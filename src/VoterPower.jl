module VoterPower

using Random
using Maybe
using Distributions

#plotting
using Plots

abstract type AbstractBallot end

struct PreferenceBallot <: AbstractBallot
    preferences::Vector{Int64} # ordered list of candidates. Should be a permutation of 1:n_candidates
end

struct InvertedPreferenceBallot <: AbstractBallot
    model::PreferenceBallot
end

InvertedPreferenceBallot(p::PreferenceBallot, params::Any) = InvertedPreferenceBallot(p) # this one doesn't need params

struct OneInversionPreferenceBallot <: AbstractBallot
    model::PreferenceBallot
    cand_to_invert::Int
end

abstract type AbstractElection end

struct Election <: AbstractElection
    n_candidates::Int
    ballots::Vector{AbstractBallot}
end

abstract type ElectionModifier end

mutable struct PermMirrors{mirror_type <: AbstractBallot} <: ElectionModifier
    perm::Vector{Int} # permutation of the voters. Just make sure the length is right.
        #if perm[i] == j, then voter i is mirrored when how_deep >= j.
    how_deep::Int # how many voters into the permutation to mirror. 0 means no mirroring.
    mirror_params::Any # parameters for the mirror type
    index_of_first_unmirrored::Int # the index of the first umirrored voter, or -1. If found, perm(index_of_first_unmirrored) == how_deep + 1.
end

# mutable struct ElectionGlass <: AbstractElection # "Glass" because it's through some arbitrary looking glass
#     model::Election
#     glass::ElectionModifier
# end
#
# ...but no, we want that type to be parametrized by the glass type, for efficient dispatch.

mutable struct ElectionGlass{G<:ElectionModifier} <: AbstractElection
    model::AbstractElection
    glass::G
end

mutable struct STV_round_record
    initial_survivors::BitVector #Technically this is redundant, but it avoids recalculating.
    initial_tallies::Vector{Float64} #copy of the weighted_tallies at the beginning of the round
    winner_found::Bool #whether a winner was found in this round
    reallocated_cand::Int #the id of the candidate who was reallocated in this round
end

mutable struct STV_record 
    n_winners::Int
    n_voters::Int
    n_candidates::Int
    glass_breadcrumbs::Any # Produced by get_breadcrumbs(election), used for now to record how_deep and index_of_first_unmirrored
    rounds::Vector{STV_round_record}
    winners::Vector{Int}
end

abstract type AbstractVoter end # Used for both voters and candidates

struct SpatialVoter <: AbstractVoter
    location::Vector{Float64}
    weights::Vector{Float64} # Diagonal of precision matrix for the voter's mahalanobis distance metric
    label::Any # eg, cluster id — for things like graphing.
end

struct Electorate #Technically we should specialize on voter type, but this is fine for now.
    voters::Vector{AbstractVoter}
    candidates::Vector{AbstractVoter}
end

## Functions for converting from electorate to election

function rate_candidate(voter::SpatialVoter, candidate::SpatialVoter)
    return exp(-0.5 * sum((voter.location - candidate.location).^2 .* voter.weights.^2))
        # Compare candidate ratings to get a preference ballot
        # Note the exp; this is a Gaussian kernel, so for ratings ballots, two faraway candidates will be rated similarly.
end

function get_ballot(voter::AbstractVoter, candidates::Vector{AbstractVoter})
    ratings = [rate_candidate(voter, candidate) for candidate in candidates]
    return PreferenceBallot(sortperm(ratings, rev=true))
end

function get_election(electorate::Electorate)
    n_candidates = length(electorate.candidates)
    ballots = [get_ballot(voter, electorate.candidates) for voter in electorate.voters]
    return Election(n_candidates, ballots)
end

## Counting voters
function count_voters(election::Election)
    return length(election.ballots)
end

function count_voters(election::ElectionGlass)
    return count_voters(election.model)
end

function count_candidates(election::Election)
    return election.n_candidates
end

function count_candidates(election::ElectionGlass)
    return count_candidates(election.model)
end

function get_quota(election::AbstractElection, n_winners::Int)
    n_voters = count_voters(election)
    return n_voters / (n_winners + 1)
end

function get_quota(record::STV_record)
    return record.n_voters / (record.n_winners + 1)
end

function get_ballot_weight(election::AbstractElection, voter_num::Int)
    return 1. # future election types may override this
end

## Breadcrumb functions

function get_breadcrumbs(election::AbstractElection)
    return nothing
end

function get_breadcrumbs(election::ElectionGlass)
    return get_breadcrumbs(election.glass)
end

function get_breadcrumbs(glass::PermMirrors)
    return (glass.how_deep, glass.index_of_first_unmirrored)
end

## Functions for the STV algorithm

function STV_record()
    return STV_record(0, 0, 0, nothing, [], []) #fill in the fields later
end

function STV_result(election::AbstractElection, n_winners::Int;

        assert_invariants::Bool=false,
        record::Union{STV_record, Nothing}=nothing,

        # Optional arguments for efficiency
        deindexify::Union{Function, Nothing}=nothing,
        indexify::Union{Function, Nothing}=nothing,
        # Optional arguments for additional outputs
        white_box_power::Union{Matrix{Float64}, Nothing}=nothing,
        distance_from_pivotality::Union{Vector{Float64}, Nothing}=nothing)
"""
    Run the Single Transferable Vote (STV) algorithm on an election.

    Includes allocating memory and initializing the running variables.



    Args:
        Election: An Election object representing the election.
        n_winners: The number of winners to elect.

        assert_invariants: A boolean indicating whether to assert invariants at the end of each round. (Explained in STV_result_inner!)
        record: A STV_record object to record the results of the election. If nothing, don't record.
            The record includes the number of winners, the number of voters, the number of candidates, the rounds of the election, and the winners.
"""
    n_voters = count_voters(election)
    n_candidates = count_candidates(election) * 2 # Each candidate has a mirror anti-candidate. 
        # The anti-candidate is the candidate with the same index, but with a negative sign.
        # If you're running a normal election, the anti-candidates will never win, so you can ignore them.

    # Create and initialize running variables
    surviving_candidates = trues(n_candidates)
    winners = zeros(Int, n_winners) # Zero is not a valid candidate index, even though negative indices are valid anti-candidates
    winners_so_far = 0
    round = 1
    ballot_piles = zeros(Int, n_candidates, n_voters)
    ballot_pile_sizes = zeros(Int, n_candidates)
    weighted_tallies = zeros(Float64, n_candidates)
    ballot_weights = ones(Float64, n_voters)

    if record != nothing
        record.n_winners = n_winners
        record.n_voters = n_voters
        record.n_candidates = n_candidates
        record.rounds = []
        record.winners = []
    end

    # Call the mutating version of the function
    result = STV_result_inner!(election, n_winners, surviving_candidates, winners, 
            winners_so_far, round, ballot_piles, ballot_pile_sizes, weighted_tallies, ballot_weights;
            assert_invariants=assert_invariants, record=record,
            deindexify=deindexify, indexify=indexify, white_box_power=white_box_power,
            distance_from_pivotality=distance_from_pivotality)
    
    if record != nothing
        record.glass_breadcrumbs = get_breadcrumbs(election)
    end

    return result
end

function get_deindexify(deindexify::Function, n_candidates::Int)
    return deindexify
end

function get_deindexify(deindexify::Nothing, n_candidates::Int)
    return cand_index_to_id(n_candidates)
end

function get_indexify(indexify::Function, n_candidates::Int)
    return indexify
end

function get_indexify(indexify::Nothing, n_candidates::Int)
    return cand_id_to_index(n_candidates)
end

function STV_result_inner!(election::AbstractElection, 
                n_winners::Int, 
                surviving_candidates::BitVector, 
                winners::Vector{Int}, 
                winners_so_far::Int, 
                round::Int, 
                ballot_piles::Matrix{Int}, 
                ballot_pile_sizes::Vector{Int}, 
                weighted_tallies::Vector{Float64}, 
                ballot_weights::Vector{Float64};

                assert_invariants::Bool=false,
                record::Union{STV_record, Nothing}=nothing,
                deindexify::Union{Function, Nothing}=nothing,
                indexify::Union{Function, Nothing}=nothing,
                white_box_power::Union{Matrix{Float64}, Nothing}=nothing,
                distance_from_pivotality::Union{Vector{Float64}, Nothing}=nothing,

                debug::Bool=false,
                )
"""
    Run the Single Transferable Vote (STV) algorithm on an election.

    The STV algorithm is a ranked-choice voting system that is used in multi-winner elections.
    It is a generalization of Instant Runoff Voting (IRV) to multi-winner elections.

    The algorithm works as follows:
    1. Each voter ranks the candidates in order of preference.
    2. The votes are tallied. If a candidate has more votes than the (Droop) quota, they are elected (in descending order of votes).
    3. If a candidate has more votes than the quota, the votes are reweighted so that they sum to the surplus, and transferred to the next preference on each ballot.
    4. If no candidate has more votes than the quota, the candidate with the fewest votes is eliminated, and their votes are transferred to the next preference on each ballot.
    5. Repeat steps 2-4 until all winners have been elected.

    Args:
        election: An Election object representing the election.
        n_winners: The number of winners to elect.
        surviving_candidates: A boolean vector indicating which candidates are still in the running. Indices run from 1 to n_candidates.
            Anti-candidates are after the candidates, in the same order. deindexify can be used to convert from candidate index to candidate id.
        winners: A vector of the candidates who have been elected so far.
        winners_so_far: The number of winners who have been elected so far.
        round: The current round of the election.
        ballot_piles: A matrix to hold the current ballot piles. Each row is a candidate, and 
            each column up to ballot_pile_sizes[row] is a (unique) ballot.
        ballot_pile_sizes: A vector indicating the sizes of the ballot_piles.
        weighted_tallies: A vector of the weighted tallies for each candidate.
        ballot_weights: A vector of the weights of each ballot.

        assert_invariants: A boolean indicating whether to assert invariants at the end of each round. Invariants include:
            - The round should be less than the number of candidates. This should be asserted at 
                the beginning of the function, and regardless of the value of assert_invariants.

            - The sum of the weighted tallies should equal the sum of the ballot weights (up to rounding).
            - The sum of the weighted tallies of the surviving candidates should equal (up to rounding) the Droop(ish) quota times 
                the number of winners remaining plus one.
            - The ballot_piles should be non-zero up to the ballot_pile_sizes, and zero thereafter.
            - The winners should be unique and non-zero up to the number of winners so far, and zero thereafter.
            - The ballot_weights should be in the range [0, 1].
            - winners_so_far should be less than or equal to min(n_winners, round).
            - the sum of ballot_pile_sizes for the surviving candidates should equal the number of ballots.
            - weighted_tallies[row] <= ballot_pile_sizes[row] for all surviving candidates.
            - (We could also check that weighted_tallies equal the sum of the appropriate ballot_weights, but that's too slow.)


"""
    #OK, let's think a bit about possible additional outputs.
    #These should be via optional mutable arguments, of course. Testing to see if arg is nothing is cheap.
    #1. Empirical ("white-box") power matrix. This is a matrix of the power of each voter to elect each winner.
    #   Dimensions: n_winners x n_voters. This is a measure of the power of each voter to elect each winner.
    
    #2. Distance from pivotality. The smallest margin by which any branch was taken. Flipping fewer than half this
    #  number of voters could not have changed the outcome. This is a measure of the robustness of the outcome, useful
    #  for speeding up binary search. (This is the simplest output that avoids at least some re-runs as we flip voters.
    #  There may be other ways to reduce redundancy, but they'd be way more complex.)

    #3. A record of the steps, in a form that allows evaluating whether any pair of ballots is "equivalently pivotal".
    #   This is already covered by STV_record, though it may be less than perfectly efficient.

    #... OK, let's add 1-2 then.

    
    # record global params
    if record != nothing
        record.n_winners = n_winners
        record.n_voters = count_voters(election)
        record.n_candidates = count_candidates(election)
        record.rounds = []
        record.winners = winners
    end

    if distance_from_pivotality != nothing
        distance_from_pivotality[1] = Inf #initialize to infinity. It's really just a number; the Vector type is for mutability.
    end


    # Calculate the Droop quota
    n_voters = count_voters(election)
    quota = n_voters / (n_winners + 1) # technically this isn't the Droop quota because we're not adding epsilon, but for the invariants above this is actually better.
    n_candidates = count_candidates(election) * 2 # Each candidate has a mirror anti-candidate. 
        # The anti-candidate is the candidate with the same index, but with a negative sign.
        # If you're running a normal election, the anti-candidates will never win, so you can ignore them.
    deindexify_ = get_deindexify(deindexify, n_candidates)#cand_index_to_id(n_candidates) # Convert from candidate index (1 to n_candidates) to candidate id (-n_candidates/2 to n_candidates/2)
        # candidate index has the anti-candidates after the candidates, in the same order.
    indexify_ = get_indexify(indexify, n_candidates) #cand_id_to_index(n_candidates) # Convert from candidate id (-n_candidates/2 to n_candidates/2) to candidate index (1 to n_candidates)

    do_initial_tallies!(election, n_winners, surviving_candidates, winners, winners_so_far, round, ballot_piles, ballot_pile_sizes, weighted_tallies, ballot_weights)
    # keep doing rounds until we have enough winners
    while winners_so_far < n_winners
        if debug
            print("Round $round\n; winners so far: $winners_so_far of $n_winners\n; surviving candidates: $(findall(surviving_candidates))\n; weighted tallies: $weighted_tallies ($(sum(weighted_tallies[surviving_candidates])))\n; ballot pile sizes: $ballot_pile_sizes\n; quota = $quota\n\n")
        end

        # Precondition assertion
        @assert round <= n_candidates

        # Record the round
        if record != nothing
            push!(record.rounds, STV_round_record(copy(surviving_candidates), copy(weighted_tallies), false, 0))
            # We'll set winner_found and reallocated_cand later
        end

        # Check if any candidate has more votes than the quota
        if maximum(weighted_tallies[surviving_candidates]) >= quota
            if distance_from_pivotality != nothing
                distance_from_pivotality[1] = minimum(distance_from_pivotality[1],
                        maximum(weighted_tallies[surviving_candidates]) - quota)
            end
                # Note `>=` ... this means we won't have a tiebreaker if the last two candidates have exactly the quota. 

            # If so, elect the candidate with the most votes
            next_survivor = argmax(weighted_tallies[surviving_candidates]) # This gets the index of the winner *among the surviving candidates*.
                # This is not the same as the candidate id.
            next_winner_index = get_survivor_index(next_survivor, surviving_candidates, n_candidates)

            if distance_from_pivotality != nothing
                second_place_tally = maximum(weighted_tallies[surviving_candidates .& (1:n_candidates .!= next_winner_index)])
                distance_from_pivotality[1] = minimum(distance_from_pivotality[1],
                        weighted_tallies[next_winner_index] - second_place_tally)
            end
            # next_winner_index = 0
            # for i in 1:n_candidates
            #     if surviving_candidates[i]
            #         next_winner_index += 1
            #         if next_winner_index == next_survivor
            #             next_winner_index = i
            #             break
            #         end
            #     end
            # end
            next_winner = deindexify_(next_winner_index)

            # Record the round
            if record != nothing
                record.rounds[end].winner_found = true
                record.rounds[end].reallocated_cand = next_winner
            end

            if white_box_power != nothing
                white_box_power[winners_so_far + 1, :] = zeros(Float64, n_voters)
                for ballot in 1:ballot_pile_sizes[next_winner_index]
                    voter_num = ballot_piles[next_winner_index, ballot]
                    white_box_power[winners_so_far + 1, voter_num] = ballot_weights[voter_num]
                end
                white_box_power[winners_so_far + 1, :] /= sum(white_box_power[winners_so_far + 1, :])
            end

            winners[winners_so_far + 1] = next_winner
            winners_so_far += 1
            surviving_candidates[next_winner_index] = false
            # Transfer the surplus votes
            surplus = weighted_tallies[next_winner_index] - quota
            weight_ratio = surplus / weighted_tallies[next_winner_index]
            #weighted_tallies[next_winner_index] = quota # Once a candidate is elected, their tally is negative.
            
            if debug
                println("Surplus = $surplus, weight ratio = $weight_ratio")
            end
            for i in 1:ballot_pile_sizes[next_winner_index]
                voter_num = ballot_piles[next_winner_index, i]
                ballot = get_ballot(election, voter_num)
                new_pile = best_surviving_preference(ballot, surviving_candidates, indexify_)
                #print("voter $voter_num, ballot $ballot, new pile $new_pile\n")
                if new_pile == 0
                    ballot_weights[voter_num] = 0.
                else
                    ballot_piles[new_pile, ballot_pile_sizes[new_pile] + 1] = voter_num
                    ballot_pile_sizes[new_pile] += 1
                    ballot_weights[voter_num] *= weight_ratio
                    weighted_tallies[new_pile] += ballot_weights[voter_num]
                end
            end
            ballot_pile_sizes[next_winner_index] = 0
            if debug
                print("\n\nFound winner $next_winner\n; winners so far: $winners_so_far of $n_winners\n; surviving_candidates = $(sum(surviving_candidates))\n\n")
            end
            round += 1
            continue
            
        end

        # Check if the candidates left are equal to the number of winners left
        if sum(surviving_candidates) == n_winners - winners_so_far
            # If so, elect the remaining candidates
            for i in 1:n_candidates
                if surviving_candidates[i]
                    next_winner = deindexify_(i)
                    winners[winners_so_far + 1] = next_winner
                    winners_so_far += 1
                    surviving_candidates[i] = false
                    if white_box_power != nothing
                        white_box_power[winners_so_far, :] = zeros(Float64, n_voters)
                        for ballot in 1:ballot_pile_sizes[i]
                            voter_num = ballot_piles[i, ballot]
                            white_box_power[winners_so_far, voter_num] = ballot_weights[voter_num]
                        end
                        white_box_power[winners_so_far, :] /= sum(white_box_power[winners_so_far, :])
                    end
                    if debug
                        print("\n\nFound winner $next_winner\n; winners so far: $winners_so_far of $n_winners\n; surviving_candidates = $(sum(surviving_candidates))\n\n")
                    end
                    round += 1
                    break
                end
            end
            continue
        end
        
        # No candidate has more votes than the quota, so eliminate the candidate with the fewest votes
        losing_survivor = argmin(weighted_tallies[surviving_candidates])
        loser_index = get_survivor_index(losing_survivor, surviving_candidates, n_candidates)
        if distance_from_pivotality != nothing
            second_to_last_tally = minimum(weighted_tallies[surviving_candidates .& (1:n_candidates .!= loser_index)])
            distance_from_pivotality[1] = minimum(distance_from_pivotality[1],
                    second_to_last_tally - weighted_tallies[loser_index])
        end
        loser = deindexify_(loser_index)

        # Record the round
        if record != nothing
            if debug
                println("Recording loser: $loser ($(weighted_tallies[surviving_candidates]) $(weighted_tallies[loser_index]))")
            end
            record.rounds[end].reallocated_cand = loser
            # winner_found is already false
        end

        surviving_candidates[loser_index] = false
        for i in 1:ballot_pile_sizes[loser_index]
            voter_num = ballot_piles[loser_index, i]
            ballot = get_ballot(election, voter_num)
            new_pile = best_surviving_preference(ballot, surviving_candidates, indexify_)
            if new_pile == 0
                ballot_weights[voter_num] = 0.
            else
                ballot_piles[new_pile, ballot_pile_sizes[new_pile] + 1] = voter_num
                ballot_pile_sizes[new_pile] += 1
                weighted_tallies[new_pile] += ballot_weights[voter_num]        
            end
        end

        # Postcondition assertion
        if assert_invariants
            @assert sum(weighted_tallies) ≈ sum(ballot_weights)
            @assert sum(weighted_tallies[surviving_candidates]) ≈ (n_winners - winners_so_far + 1) * quota
            @assert all(ballot_piles[1:n_candidates, 1:maximum(ballot_pile_sizes)] .!= 0)
            @assert all(ballot_piles[1:n_candidates, maximum(ballot_pile_sizes)+1:end] .== 0)
            @assert all(winners[1:winners_so_far] .!= 0)
            @assert all(winners[winners_so_far+1:end] .== 0)
            @assert all(0 .<= ballot_weights .<= 1)
            @assert winners_so_far ≤ min(n_winners, round)
            @assert sum(ballot_pile_sizes[surviving_candidates]) == n_voters
            @assert all(weighted_tallies[surviving_candidates] .<= ballot_pile_sizes[surviving_candidates])
        end

        if debug
            print("\n\nRound $round done\n; winners so far: $winners_so_far of $n_winners\n; surviving_candidates = $(sum(surviving_candidates))\n\n")
        end
        round += 1
    end

    #record winners
    if record != nothing
        record.winners = winners
    end

    return winners
end

# Helper functions

function do_initial_tallies!(election::AbstractElection, 
                n_winners::Int, 
                surviving_candidates::BitVector, 
                winners::Vector{Int}, 
                winners_so_far::Int, 
                round::Int, 
                ballot_piles::Matrix{Int}, 
                ballot_pile_sizes::Vector{Int}, 
                weighted_tallies::Vector{Float64}, 
                ballot_weights::Vector{Float64};

                debug::Bool=false)
    n_voters = count_voters(election)
    n_candidates = count_candidates(election) * 2 # Each candidate has a mirror anti-candidate. 
        # The anti-candidate is the candidate with the same index, but with a negative sign.
        # If you're running a normal election, the anti-candidates will never win, so you can ignore them.
    indexify = cand_id_to_index(n_candidates) # Convert from candidate id (-n_candidates/2 to n_candidates/2) to candidate index (1 to n_candidates)
    qq = indexify(4)
    if debug
        println("n_candidates = $n_candidates; indexify = $(qq)")
    end

    # Initialize the ballot piles
    for i in 1:n_candidates
        ballot_pile_sizes[i] = 0
    end
    for i in 1:n_voters
        ballot = get_ballot(election, i)
        pile = best_surviving_preference(ballot, surviving_candidates, indexify)
        if pile == 0
            ballot_weights[i] = 0.
        else
            ballot_piles[pile, ballot_pile_sizes[pile] + 1] = i
            ballot_pile_sizes[pile] += 1
        end
    end

    # Initialize the weighted tallies
    for i in 1:n_candidates
        weighted_tallies[i] = 0.
    end
    for i in 1:n_candidates
        if surviving_candidates[i]
            for j in 1:ballot_pile_sizes[i]
                voter_num = ballot_piles[i, j]
                weighted_tallies[i] += ballot_weights[voter_num]
            end
        end
    end
end

function get_survivor_index(survivor::Int, surviving_candidates::BitVector, n_candidates::Int)
    survivors_so_far = 0
    for i in 1:n_candidates
        if surviving_candidates[i]
            survivors_so_far += 1
            if survivors_so_far == survivor
                return i
            end
        end
    end
end

function get_ballot(election::Election, voter_num::Int)
    return election.ballots[voter_num]
end

function get_ballot(election::ElectionGlass, voter_num::Int) # no bang because I'm lazy and the side effect is just for efficiency
    return view_ballot!(get_ballot(election.model, voter_num), election.glass, voter_num)
end

function view_ballot!(ballot::PreferenceBallot, glass::PermMirrors{mirror_type}, voter_num::Int) where {mirror_type <: AbstractBallot}
    """
    View a ballot through a glass (permutation mirrors).

    Args:
        ballot: A PreferenceBallot object representing the ballot.
        glass: A PermMirrors object representing the glass.
        voter_num: The number of the voter.

    side effect: updates glass.index_of_first_unmirrored if the voter is the first unmirrored voter. (thus the bang)

    Returns:
        The ballot, viewed through the glass.
    """
    place_in_perm = glass.perm[voter_num]
    if place_in_perm <= glass.how_deep
        return mirror_type(ballot, glass.mirror_params)
    else
        if place_in_perm == glass.how_deep + 1
            glass.index_of_first_unmirrored = voter_num
        end
        return ballot
    end
end

function get_mirror_of_by(ballot::AbstractBallot, glass::ElectionGlass)
    """
    Get the mirror of a ballot through a glass. Follows the mirroring rules of the glass (InvertedPreferenceBallot, OneInversionPreferenceBallot, etc.)
    """
    return get_mirror_of_by(ballot, glass.glass)
end

function get_mirror_of_by(ballot::AbstractBallot, glass::PermMirrors{mirror_type}) where {mirror_type <: AbstractBallot}
    return mirror_type(ballot, glass.mirror_params)
end

function get_mirror_of_by(ballot::mirror_type, glass::PermMirrors{mirror_type}) where {mirror_type <: AbstractBallot}# two mirrors of the same type cancel out
    # Technically two OneInversionPreferenceBallots could have different cand_to_invert, but we're not going to worry about that.
    return ballot.model
end


function best_surviving_preference(ballot::PreferenceBallot, surviving_candidates::BitVector, indexify::Function;
        dont_invert_ballot::Bool=true)
"""
    Find the highest-ranked candidate who is still in the running.

    Args:
        ballot: A PreferenceBallot object representing the ballot.
        surviving_candidates: A boolean vector indicating which candidates are still in the running. Indices run from 1 to n_candidates.
            Anti-candidates are after the candidates, in the same order. deindexify can be used to convert from candidate index to candidate id.
        n_candidates: The number of candidates in the election.
        indexify: A function to convert from candidate id to candidate index.

    Returns:
        The index of the highest-ranked candidate who is still in the running, or 0 if no such candidate exists.

    Note: if a ballot includes zeros, this (incorrectly) puts anti-candidates with preferences above ignored candidates and ignored anti-candidates.
"""
    for candidate in ballot.preferences
        #c = indexify(candidate)
        if dont_invert_ballot
            c = indexify(candidate)
        else
            c = indexify(-candidate)
        end
        if c > length(surviving_candidates)
            println("$(indexify(-1))")
            println("c = $c, candidate = $candidate, surviving_candidates = $surviving_candidates, prefs = $(ballot.preferences)")
        end
        if c!=0 && surviving_candidates[c]
            return c 
        end
    end
    # Now do the anti-candidates
    for candidate in ballot.preferences[end:-1:1]
        if dont_invert_ballot
            c = indexify(-candidate)
        else
            c = indexify(candidate)
        end
        if c!=0 && surviving_candidates[c]
            return c 
        end
    end
    return 0
end

function best_surviving_preference(ballot::InvertedPreferenceBallot, surviving_candidates::BitVector, indexify::Function)
    return best_surviving_preference(ballot.model, surviving_candidates, indexify; dont_invert_ballot=false)
end

function best_surviving_preference(ballot::OneInversionPreferenceBallot, surviving_candidates::BitVector, n_candidates::Int, indexify::Function)
    for candidate in ballot.model.preferences
        if candidate == ballot.cand_to_invert
            c = indexify(-candidate)
        else
            c = indexify(candidate)
        end
        if c!=0 && surviving_candidates[c]
            return c 
        end
    end
    # Now do the anti-candidates
    for candidate in ballot.model.preferences[end:-1:1]
        if candidate == ballot.cand_to_invert
            c = indexify(candidate)
        else
            c = indexify(-candidate)
        end
        if c!=0 && surviving_candidates[c]
            return c 
        end
    end
    return 0
end

function cand_id_to_index(n_candidates::Int)
    return function(cand_id::Int)
        if cand_id > 0
            return cand_id
        else
            return (n_candidates ÷ 2) - cand_id
        end
    end
end

function cand_index_to_id(n_candidates::Int)
    return function(cand_index::Int)
        if cand_index <= (n_candidates ÷ 2)
            return cand_index
        else
            return Int64((n_candidates ÷ 2) - cand_index)
        end
    end
end


# Allow reusing the same memory for running variables

function reinitialize_running_variables!(election::Election, n_winners::Int, surviving_candidates::BitVector, winners::Vector{Int}, winners_so_far::Int, round::Int, ballot_piles::Matrix{Int}, ballot_pile_sizes::Vector{Int}, weighted_tallies::Vector{Float64}, ballot_weights::Vector{Float64})
    n_candidates = election.n_candidates * 2 # Each candidate has a mirror anti-candidate. 
        # The anti-candidate is the candidate with the same index, but with a negative sign.
        # If you're running a normal election, the anti-candidates will never win, so you can ignore them.
    for i in 1:n_candidates
        surviving_candidates[i] = true
    end
    for i in 1:n_winners
        winners[i] = 0
    end
    winners_so_far = 0
    round = 1
    for i in 1:n_candidates
        ballot_pile_sizes[i] = 0
    end
    for i in 1:length(ballot_weights)
        ballot_weights[i] = 1.
    end
    do_initial_tallies!(election, n_winners, surviving_candidates, winners, winners_so_far, round, ballot_piles, ballot_pile_sizes, weighted_tallies, ballot_weights)
end

function STV_result!(election::Election, 
                n_winners::Int, 
                surviving_candidates::BitVector, 
                winners::Vector{Int}, 
                winners_so_far::Int, 
                round::Int, 
                ballot_piles::Matrix{Int}, 
                ballot_pile_sizes::Vector{Int}, 
                weighted_tallies::Vector{Float64}, 
                ballot_weights::Vector{Float64};
                
                assert_invariants::Bool=false,
                record::Union{STV_record, Nothing}=nothing,
                deindexify::Union{Function, Nothing}=nothing,
                indexify::Union{Function, Nothing}=nothing,
                white_box_power::Union{Matrix{Float64}, Nothing}=nothing,
                distance_from_pivotality::Union{Vector{Float64}, Nothing}=nothing)
    reinitialize_running_variables!(election, n_winners, surviving_candidates, winners, winners_so_far, round, ballot_piles, ballot_pile_sizes, weighted_tallies, ballot_weights)
    return STV_result_inner!(election, n_winners, surviving_candidates, winners, winners_so_far, 
            round, ballot_piles, ballot_pile_sizes, weighted_tallies, ballot_weights;
            assert_invariants=assert_invariants, record=record,
            deindexify=deindexify, indexify=indexify, white_box_power=white_box_power, 
            distance_from_pivotality=distance_from_pivotality)
end

# Functions to create random voters and elections

function random_preference_ballot(n_candidates::Int)
    return PreferenceBallot(randperm(n_candidates))
end

function random_STV_election(n_candidates::Int, n_voters::Int)
    ballots = [PreferenceBallot(randperm(n_candidates)) for i in 1:n_voters]
    return Election(n_candidates, ballots)
end

function random_preference_ballot(n_candidates::Int, prefers::Int)
"""
    Create a random preference ballot where the voter prefers a specific candidate.
"""
    remaining_prefs = setdiff(1:n_candidates, [prefers])
    return PreferenceBallot([prefers; remaining_prefs[randperm(n_candidates-1)]])
end

function random_STV_election(n_candidates::Int, n_voters::Int, n_triangle_groups::Int)
"""
    Create a random STV election with a few groups, with relative sizes 1, 2, 3, ..., n_triangle_groups.
    Each group should prefer the candidate with the same index. The remaining candidates are 
    randomly ordered ONCE for the entire group.
"""
    @assert n_candidates >= n_triangle_groups
    total_triangle_size = n_triangle_groups * (n_triangle_groups + 1) // 2
    size_base = n_voters ÷ total_triangle_size
    size_remainder = n_voters - (total_triangle_size * size_base)
    ballots = []
    for i in 1:n_triangle_groups
        ballot = random_preference_ballot(n_candidates, i)
        for j in 1:(size_base * i)
            push!(ballots, ballot)
        end
    end
    for i in 1:size_remainder
        # Draw a ballot from the existing ballots
        ballot = ballots[rand(1:length(ballots))]
        push!(ballots, ballot)
    end
    return Election(n_candidates, ballots)
end

# Functions for creating electorates of spatial voters 

# Three possible ways to create a spatial electorate:
# 1. Gaussian model of voters and candidates
# 2. Dirichlet mixture model of (Gaussian clusters of) voters and candidates
# 3. Hierarchical Dirichlet mixture model of voters and candidates. 
#     That is to say, a tree, where each branch is a DMM and each leaf is a GM.

# Option 1 is probably too simple, as it makes systemic Condorcet cycles impossible.
# Option 3 is probably overkill — not in terms of the electorates it produces, but just 
# in terms of the complexity to code the model. Basically, you'd need separate propagation and cutoff rules
# for alpha, precision (inverse variance of GMMs), and dimension weights — and all of those would have parameters
# that would need to be tuned (either a priori or empirically).

# So let's code options 1 (as a simple example, and as a building block) and 2 (as a more realistic model).
# That means that even though 2 could be seen as a special case of 3, we'll make types for 2 separately, 
# then maybe do 3 later without worrying about the code/type duplication.

# Still, even for option 2 (DMM), we'll need to decide how to propagate precision and weights. Sub-options:
# A. Just use the same precision and weights for all subclusters. This is the simplest, but not realistic.
#    In real life, different voter groups care about different issues differently.
# B. Propagate precision and weights from the parent cluster to the subclusters, but allow them to change.
#    But how? Let's see...:
#    * Precision and weight on a given dimension are correlated. A group that cares a lot about a dimension
#      (high weight) will also vary less on that dimension (high precision).
#    * This will also correlate with squared deviance on that dimension. (Or is it better to use absolute deviance?)
#    * But maybe we want a ceiling on the weight * deviance product, because otherwise we'll get groups that
#      both extreme and picky, making our a priori dimension weights (and thus, the calculation of the effective
#      number of dimensions of the electorate as a whole, and the dimension weight cutoffs) mean something quite 
#      different from their naive interpretation.
#    * I think the lesson here is that, assuming we're propagating using chi squared distributions, we should
#      make sure the degrees of freedom aren't too low.
# ...OK, so "simple" version of B is:
# For a new cluster, draw standard normals A, B, C, D for each dimension. Position is A, precision (factor) is
# sqrt(A²+B²+C²/3), and weight factor is sqrt(A²+B²+D²/3). (Thus, precision and weight are both from "chi" distributions with 3 df.)

# Let's code that up. If we want to vary df, we can refactor later.

abstract type SpatialGenerator end

struct GaussianSpatialGenerator <: SpatialGenerator
    center::Vector{Float64}
    precision::Vector{Float64} # sqrt diagonal of precision matrix (inverse stdev)
    weights::Vector{Float64} # dimension weights (diagonal of precision for mahalanobis distance metric)
end

mutable struct DirichletSpatialGenerator <: SpatialGenerator
    subclusters::Vector{GaussianSpatialGenerator}
    cluster_sizes::Vector{Float64} #in practice, ints; but since we'll be adding alpha in use, store 'em as floats 
    dim_weights::Vector{Float64}
    dim_weight_factor::Float64 # for auditing only — once we have the weights, this is redundant
    dim_weight_cutoff::Float64 # for auditing only — once we have the weights, this is redundant
    alpha::Float64
    precision_factor::Float64 # cluster precision is sqrt(chi²/df) * precision_factor * master_precision (master_precision is 1 for non-hierarchical)
    # DF & correlations for propagating precision and weight heirarchically would go here. Not sure how I'd even parametrize that.
end


function DirichletSpatialGenerator(dim_weight_factor::Float64, dim_weight_cutoff::Float64, alpha::Float64, precision_factor::Float64)
    """
    Set up a Dirichlet mixture model of spatial voters. Later we can draw from it to get voters and candidates
    for an electorate.
    """
    @assert 0 < dim_weight_factor < 1
    @assert 0 < dim_weight_cutoff < 1
    @assert alpha > 0

    dim_weights = [1.]
    df = 1.5
    while true
        next_dim_weight = (dim_weights[end] * dim_weight_factor * 
            # chi-squared distribution with df degrees of freedom, normalized
            rand(Chisq(df)) / df)
        if (next_dim_weight / sum(dim_weights)) < dim_weight_cutoff # note that d_w_c is in terms of odds
            break
        end
        push!(dim_weights, next_dim_weight)
    end
    return DirichletSpatialGenerator([], [], dim_weights, dim_weight_factor, dim_weight_cutoff, precision_factor, alpha)

end

function draw_voter!(generator::GaussianSpatialGenerator, label::Any) # bang because generally this function could modify the generator, even though this version doesn't
    return SpatialVoter(generator.center + rand(Normal(0., 1.), length(generator.center)) ./ (generator.precision),
        generator.weights, label)
end

function draw_voter!(generator::GaussianSpatialGenerator)
    return draw_voter!(generator, nothing)
end

function draw_voter!(generator::DirichletSpatialGenerator)
    probs = vcat(generator.cluster_sizes, [generator.alpha])
    subcluster = rand(Categorical(probs / sum(probs)))
    if subcluster > length(generator.subclusters)
        @assert subcluster == length(generator.subclusters) + 1
        # Draw a new subcluster
        rands = randn(length(generator.dim_weights), 4) # 4 rands for A, B, C, D 
        center = rands[:, 1]
        precision = (sum(rands[:, 1:3].^2, dims=2) * generator.precision_factor / 3)
        weights = sqrt.(sum(rands[:, [1, 2, 4]].^2, dims=2) / 3) .* generator.dim_weights
        println("New subcluster: center $center, precision $precision, weights $weights")
        push!(generator.subclusters, GaussianSpatialGenerator(center, vec(precision), vec(weights)))
        push!(generator.cluster_sizes, 1.)
    else
        generator.cluster_sizes[subcluster] += 1.
    end
    return draw_voter!(generator.subclusters[subcluster], subcluster)
end

function draw_electorate!(generator::SpatialGenerator, n_voters::Int, n_candidates::Int)
    voters = [draw_voter!(generator) for i in 1:n_voters]
    candidates = [draw_voter!(generator) for i in 1:n_candidates]
    return Electorate(voters, candidates)
end

function plot_electorate(electorate::Electorate)
    # Plot the first two dimensions of the voters, color coded by cluster
    scatter([voter.location[1] for voter in electorate.voters], [voter.location[2] for voter in electorate.voters], 
        group=[voter.label for voter in electorate.voters],
        title="Voters",
        xlabel="Dimension 1",
        ylabel="Dimension 2 $(mean([voter.weights[2] for voter in electorate.voters]))",
        # semitransparent
        markersize=3, markerstrokewidth=0, markeralpha=0.1)

end

# Functions for getting voter power (in the sense of black-box or mirror-universe power). For now let's call this BBVP.

# BBVP is defined as the probability that a voter is pivotal for a given candidate, under the "mirror universe distribution"
# of elections, minus the probability that they're anti-pivotal, times the number of voters. By construction, the sum of BBVPs
# for a given candidate is 1. (This is because basic assumptions about the voting system imply that a winner of the actual
# election will not win a fully-mirrored election; thus, for any permutation, the number of pivotal voters minus the number of
# anti-pivotal voters is 1.)

# To generate the mirror universe distribution,
# 1. Take the actual election as it occurred.
# 2. Take a random permutation of the voters.
# 3. Take a uniformly random fraction of the voters and mirror up to that fraction in order of the permutation.

# A voter is pivotal under this distribution if they are the next voter to be mirrored, AND if mirroring them would cause
# the given candidate not to win. (If the candidate is already not winning, the voter is not pivotal.)
# Anti-pivotality is defined similarly, but causing them to win.

# (OK this definition is problematic because the draw from the mirror universe distribution yields not just an election, but
# also a "next voter" to check. Pretty sure it's possible to rewrite this to something clean and equivalent but that's not
# the point right now.)

# Estimating a given voter's BBVP by naive brute-force sampling is infeasible (it's O((V+1)!), where V is the number of 
# voters) so we need to take advantage of
# symmetries. In particular, when we use a specific permutation to find a pivotal voter, we can find all voters who are
# equivalent to that voter in terms of pivotality; that is, who would be pivotal under a permutation that swaps them with
# the pivotal voter. This will generally be approximately V/C voters, where C is the number of candidates. This speeds things
# up by a factor of (almost) V/C because we're effectively finding pivotality for V/C different permutations at once.

# (Actually, "equivalency" here could be defined in two ways: either the simple definition above, or "weighted equivalency",
# where we look for voters who are "pulling in the same direction", then weight by "how far they're pulling". These are
# asymptotically the same; "weighted equivalency" is a bit more efficient, but "simple equivalency" is slightly more correct 
# in small samples.)

# Still, we're never going to exhaustively search all permutations, so any measure of BBVP will be an estimate (with an estimated
# error). We can reduce the error by taking more samples, but that's about it.

# So the key functions will be:
# sample_a_pivotal_voter_for: find *a* pivotal voter for one random permutation. Note that this uses binary search, so it's
#      essentially assuming that there's no anti-pivotal voters (thus, only one pivotal voter). 
#      This assumption should almost entirely hold for STV. Insofar as it doesn't,
#      this will bias things slightly in a way I haven't worked out yet.
# find_equivalent_voters: find all voters who are equivalent to a given voter in terms of pivotality.
#      (takes a flag for "weighted equivalency" or not)
# sample_one_BBVP: use the two above to calculate the BBVP contribution for that one permutation and equivalents.
#      (takes a flag for "weighted equivalency" or not)
# estimate_BBVP: get estimated_BBVP (and its error) for all voters for a given candidate, by looping over sample_one_BBVP and averaging.
# estimate_all_BBVPs: get estimate_BBVPs for all winning candidates (in a matrix)

function sample_a_pivotal_voter_for(election::Election, n_winners::Int, winner_index::Int, get_winners::Function, mirror_type::Type; # mirror_type <: AbstractBallot
    debug::Bool=false)
    """
    Given an election and a (mirror-free) winner of that election, use a permutation of the voters to find 
    a voter who is pivotal for that winner as you mirror the voters in that permuted order.

    Uses binary search
    """
    n_voters = length(election.ballots)
    
    # Make an ElectionGlass of the election, with a random permutation
    perm = randperm(n_voters)
    glass = PermMirrors{mirror_type}(perm, n_voters ÷ 2, winner_index, -1) #technically for mirror_type==InvertedPreferenceBallot, winner_index could be Nothing, but it doesn't hurt to keep it as an Int
    election_glass = ElectionGlass(election, glass)
    record_lo = STV_record()
    record_hi = STV_record()
    record_mid = STV_record()

    min_is_elected = 0
    max_isnt_elected = n_voters
    possible_pivot = -1
    while max_isnt_elected - min_is_elected > 1
        mid = (max_isnt_elected + min_is_elected) ÷ 2
        # set the depth of the glass
        glass.how_deep = mid
        #clear index_of_first_unmirrored
        glass.index_of_first_unmirrored = -1
        winners = get_winners(election_glass, n_winners, record=record_mid) #side effect: sets index_of_first_unmirrored
        @assert glass.index_of_first_unmirrored != -1
        if debug
            inverse_perm = invperm(glass.perm)
            perm_range = max(1,glass.how_deep - 1):min(length(election.ballots), glass.how_deep + 1)
            println("triangulating to unmirrored: $mid: $(glass.index_of_first_unmirrored) perm range = $(inverse_perm[perm_range]) $(perm_range) $(findall(x->x==glass.index_of_first_unmirrored, inverse_perm))")
            println("breadcrumbs: $(record_mid.glass_breadcrumbs)")
        end
        if !(winner_index in winners)
            if debug
                println("Winner not found at depth $mid ($min_is_elected - $max_isnt_elected, winner = $winner_index) $(winners)")
                println("first unmirrored ballot: $(get_ballot(election, glass.index_of_first_unmirrored))")
            end
            max_isnt_elected = mid
            record_hi, record_mid = record_mid, record_hi #leave record_mid "free" for the next iteration; hi and lo should be what they claim.
        else
            if debug
                println("Winner found at depth $mid ($min_is_elected - $max_isnt_elected, winner = $winner_index) $(winners)")
                println("first unmirrored ballot: $(get_ballot(election, glass.index_of_first_unmirrored)) for winner $(winner_index)")
            end
            min_is_elected = mid
            possible_pivot = glass.index_of_first_unmirrored
            record_lo, record_mid = record_mid, record_lo #leave record_mid "free" for the next iteration; hi and lo should be what they claim.
        end
    end

    if possible_pivot == -1
        println("possible_pivot == -1; min_is_elected = $min_is_elected; max_isnt_elected = $max_isnt_elected")
        if min_is_elected == 0
            # We have to rerun, not to get winners, but to get record_lo
            glass.how_deep = 0
            winners = get_winners(election_glass, n_winners, record=record_lo)
        else
            throw("min_is_elected != 0 but possible_pivot == -1")
        end
    else
        if debug
            println("lo breadcrumbs: $(record_lo.glass_breadcrumbs), min_is_elected = $min_is_elected")
        end
    end
    if record_hi.n_candidates == 0 # should be impossible -- would mean winner was still elected when all but one voters had flipped
        throw("record_hi.n_candidates == 0")
    end
    glass.how_deep = min_is_elected
    return (possible_pivot, election_glass, record_lo, record_hi) # return the glass so we can find equivalent voters
end

mutable struct pivot_record # This only makes sense in the context of (the arguments of) a call to find_pivotal_rounds. 
    round::Int
    margin::Float64 # The "margin of decision" for this round. Could be 0. for a tie.
        # note that margin must be less than or equal to ballot weight for that round, EXCEPT that in the symmetry-breaking round, if the ballot is pivotal in both directions, it
        # could be up to twice that.
    ballot_pile::Int
end

function find_pivotal_rounds(voter_num::Int, election_glass::ElectionGlass, pre_pivot_record::STV_record, post_pivot_record::STV_record,
    winner_index::Int,
    quota::Float64, indexify_::Function;
    
    debug::Bool=false)
    """
    Given a voter number, an ElectionGlass (just pre-pivot), and STV_records for the pre-pivot and post-pivot elections,
    find the rounds in which the voter is pivotal, and the remaining weight of the voter in those rounds.
    
    Algorithm:
        1. Find the round in the pre election in which the given winner is elected.
        2. Find the flow trees of how they got their votes, in the pre and post elections.
        3. Find the branch(es) where they got fewer votes.
        4. Find where the given voter was pivotal in those branches.
        5. Record those pivot points in order, adding up the lost votes for the given winner, until the winner's pile has been
            reduced below a quota.

    Returns:
        pivotal_round_surviving_candidates: A vector of vectors of surviving candidates in each pivotal round.
        pre_ballot_pile_and_weight: A vector of tuples of the ballot pile and weight of the pre ballot in each pivotal round.
        post_ballot_pile_and_weight: A vector of tuples of the ballot pile and weight of the post ballot in each pivotal round.

    Note: This function is not optimized for speed. It's optimized for clarity and correctness. It's not expected to be a bottleneck.
    """
    if debug
        println("Finding pivotal rounds for voter $voter_num on winner $winner_index")
    end
    n_candidates = count_candidates(election_glass.model)
    n_voters = count_voters(election_glass.model)
    winning_round = -1
    for i in 1:length(pre_pivot_record.rounds)
        if winner_index == pre_pivot_record.rounds[i].reallocated_cand
            winning_round = i
            break
        end
    end
    @assert (winning_round != -1) #"Winner not found in pre-pivot record"

    if debug
        println("Winner found in round $winning_round")
    end

    later_donors = BitVector((i==winner_index) for i in 1:n_voters)
    found_equiv_round = 1

    # find the two-way set difference of winners in the two records. Note that winners is an array of ints, not a BitVector.
    # so, not "different_winners = pre_pivot_record.winners .!= post_pivot_record.winners"
    different_winners = setdiff(Set(pre_pivot_record.winners), Set(post_pivot_record.winners))
    different_winners_bitvector = BitVector((i in [indexify_(w) for w in different_winners]) for i in 1:n_candidates)

    # Look for rounds in post_pivot_record where the winner's tally is the same as this round in pre_pivot_record
    for j in length(post_pivot_record.rounds):-1:2 #iterate backwards from decisive round
                        # Don't bother checking round 1 because it's "equivalent" by construction. 
        # Look for an "equivalent" round in the post election.
        # This logic does not find all types of "equivalencies" because that would be too slow and too hard to code. 
        # So we're just looking for a round where the set of survivors is the same, and all the different_winners still survive. 
        # That is to say, while the earlier 
        # rounds may have eliminated candidates in a different order,
        # things have ended up in the same place, modulo the pivotal voter.
        # I think this is immune from false positives?
        # As for false negatives, this could miss "equivalencies" where dead-end changes in elimination order are interleaved
        # with pivotal ones, instead of the dead-end stuff all happening up-front; 
        # but false negatives on round equivalency are only a concern for efficiency, not
        # correctness.
        if ((pre_pivot_record.rounds[j].initial_survivors == post_pivot_record.rounds[j].initial_survivors) &
                all((!different_winners_bitvector[i] | pre_pivot_record.rounds[j].initial_survivors[i]) for i in 1:n_candidates))
            found_equiv_round = j
            break
        end
    end

    if debug
        println("""Found equivalent round $found_equiv_round
                pre $(pre_pivot_record.rounds[found_equiv_round].initial_survivors), post $(post_pivot_record.rounds[found_equiv_round].initial_survivors),
                pre $(pre_pivot_record.rounds[found_equiv_round].initial_tallies), post $(post_pivot_record.rounds[found_equiv_round].initial_tallies), 
                sum abs diff $(sum(abs.(pre_pivot_record.rounds[found_equiv_round].initial_tallies - post_pivot_record.rounds[found_equiv_round].initial_tallies)))""")
    end


    # OK, now we look for rounds between found_equiv_round and winning_round where the given voter is pivotal at the round level.
    
    # Now, false positives are possible (?): cases where the voter is pivotal at the round level, but not at the election level.
    # (Most of those would correspond to false negatives in the "equivalency" logic above, but not all.)
    # I can't think of any logic that would avoid this, so we'll just have to live with it. I think it's a small/unlikely effect (maybe impossible?)


    # ...
    # So what's the return value here? For each of pre and post, we need a list of pivot_records, each of which contains:
    # 1. A pivotal round
    # 2. A margin
    # 3. A ballot pile for the pivotal ballot
    
    # Note that the first pivotal round is the last equivalent round, because the voter must be how equivalency is broken. BUT this is a special case:
    # In this last equivalent round, the voter may be pivotal not just as a pre- or post-ballot, but as both at once.

    pre_pivots = Vector{pivot_record}() # we know the voter is pivotal in the last equivalent round, because the next round is different.
    post_pivots = Vector{pivot_record}() # we know the voter is pivotal in the last equivalent round, because the next round is different.
    current_pre_weight = [get_ballot_weight(election_glass.model, voter_num)]
    current_post_weight = [-1.] # this will be overwritten in found_equiv_round

    base_ballot = get_ballot(election_glass, voter_num)
    post_base_ballot = get_mirror_of_by(base_ballot, election_glass.glass)
    current_pre_weight_by_pivotal_round = Vector{Float64}()
    pre_ballot_pile_by_pivotal_round = Vector{Int}()
    for i in 1:winning_round
        if i == found_equiv_round
            current_post_weight = [current_pre_weight[1]]
        end
        if debug
            println("Round $i")
        end
        pre_round = pre_pivot_record.rounds[i]
        post_round = post_pivot_record.rounds[i]
        ballot_pile = best_surviving_preference(base_ballot, pre_round.initial_survivors, indexify_)
        post_ballot_pile = best_surviving_preference(post_base_ballot, post_round.initial_survivors, indexify_)
        same_init = pre_round.initial_survivors == post_round.initial_survivors
        if ballot_pile == 0
            # The voter's ballot is not in the surviving candidates
            if debug
                println("   Voter $voter_num has no surviving candidates")
            end
            continue
        end
        in_runup = i < found_equiv_round
        pre_pivot = check_round_pivotality!(pre_round, ballot_pile, current_pre_weight, quota, i==found_equiv_round, in_runup, i, indexify_; debug=debug)
        if !in_runup
            post_pivot = check_round_pivotality!(post_round, post_ballot_pile, current_post_weight, quota, i==found_equiv_round, false, i, indexify_; debug=debug)
            
            if i == found_equiv_round
                num_pivots = sum((pre_pivot != nothing, post_pivot != nothing))
                if num_pivots == 0
                    raise("Voter $voter_num changes round $i outcome, but apparently not pivotal in either direction???")
                elseif num_pivots == 1
                    pre_pivot = filter_pivot(pre_pivot, current_post_weight[1])
                    post_pivot = filter_pivot(post_pivot, current_pre_weight[1])
                end
            end
            append_pivot!(pre_pivots, pre_pivot)
            append_pivot!(post_pivots, post_pivot)
        end




    end
    return (pre_pivots, post_pivots)
end

function check_round_pivotality!(round_record, ballot_pile::Int, current_weight::Vector{Float64}, quota::Float64, was_equiv::Bool, reweight_only::Bool, round_num::Int, indexify_::Function; debug::Bool=false)
    """
    Check if a voter is pivotal in a given round. Also, update their weight for the next round.

    returns: Union{Nothing, pivot_record}
    """
    result::Union{Nothing, pivot_record} = nothing
    if was_equiv
        margin_fac = .5
    else
        margin_fac = 1.
    end
    if round_record.winner_found
        reallocated_cand = indexify_(round_record.reallocated_cand)
        if debug 
            println("   Winner found: $(reallocated_cand); ballot pile $ballot_pile; survivors $(round_record.initial_survivors)")
        end
        if ballot_pile == reallocated_cand
            # The voter's ballot is in the winning candidate's pile
            
            # Thus, the ballot is pivotal iff the difference between the winning tally and the 
            # second-place tally is less than the voter's weight, OR if those are equal and the 
            # second-place index is less than the winner's index.
            second_place_tally = maximum(round_record.initial_tallies[
                round_record.initial_survivors .& (1:length(round_record.initial_survivors) .!= reallocated_cand)])

            if debug
                println("   Voter $voter_num is in the winning candidate's pile")
                println("   ..key tallies: $second_place_tally, $(round_record.initial_tallies[reallocated_cand]) ($current_weight) .... [$(round_record.initial_tallies[round_record.initial_survivors])]")
            end
            if !reweight_only
                margin = (round_record.initial_tallies[reallocated_cand] - second_place_tally) * margin_fac
                    #for rounds where the voter may be pivotal in both directions, margin_fac is .5 to allow for that. filter_pivot fixes that on the way out if it turns out only pivotal one way.
                if margin < current_weight[1]
                    result = pivot_record(round_num, margin, ballot_pile)
                    if debug
                        println("   Voter $voter_num is pivotal in round $i: weight $current_weight, second place $second_place_tally, winner tally $(round_record.initial_tallies[reallocated_cand])")
                    end
                    return result
                elseif margin == current_weight[1]
                    second_place_index = argmax(round_record.initial_tallies[
                        round_record.initial_survivors .& (1:length(round_record.initial_survivors) .!= reallocated_cand)])
                    if second_place_index < reallocated_cand
                        result = pivot_record(round_num, margin * margin_fac, ballot_pile)
                        if debug
                            println("   Voter $voter_num is barely pivotal in round $i: weight $current_weight, second place $second_place_tally, winner tally $(round_record.initial_tallies[reallocated_cand])")
                        end
                        return result
                    end
                end
            end #!reweight_only

            #reweight for next round
            winner_tally = round_record.initial_tallies[reallocated_cand]
            current_weight[1] *= (winner_tally - quota) / winner_tally
        end


    else # not round_record.winner_found, so elimination round
        if reweight_only
            if debug 
                println("   Skipping elim because round < found_equiv_round")
            end
            return result #nothing
        end
        #the ballot is pivotal if
        #   * the voter's ballot is in the second-to-last pile AND
        #       * the difference between the second-to-last tally and the last tally is less than the voter's weight OR
        #       * those are equal and the last index is less than the second-to-last index
        #Note that in principal it could be pivotal from a bigger-than-second-to-last pile, 
        # but that requires two or more improbably tight margins in a row. 
        # We'll ignore that unless/until it causes a test or assertion to fail.
        reallocated_cand = indexify_(round_record.reallocated_cand)
        not_last_survivors = round_record.initial_survivors .& (1:length(round_record.initial_survivors) .!= reallocated_cand)
        if sum(not_last_survivors) <= 1
            # The voter's ballot is in the last pile. It can't be pivotal.
            if debug
                println("   Voter $voter_num is in the last pile")
            end
            return result #nothing
        end
        surviving_second_to_last = argmin(round_record.initial_tallies[
            not_last_survivors])
        second_to_last = get_survivor_index(surviving_second_to_last, not_last_survivors, length(round_record.initial_survivors))
        if debug
            println("   Second to last: $second_to_last ($(round_record.initial_tallies[second_to_last])) [$(round_record.reallocated_cand) ($(round_record.initial_tallies[reallocated_cand]))]; ballot pile $ballot_pile; survivors $(round_record.initial_survivors); tallies $(round_record.initial_tallies[round_record.initial_survivors])")
            println("   post: [$(post_round.initial_tallies[round_record.initial_survivors]) $((post_round.initial_survivors))]")
        end
        if ballot_pile == second_to_last
            second_to_last_tally = round_record.initial_tallies[second_to_last]
            if debug
                println("   key tallies: $second_to_last_tally, $(round_record.initial_tallies[reallocated_cand]) ($current_weight[1]) .... [$(round_record.initial_tallies[round_record.initial_survivors])]")
                println("   post: [$(post_round.initial_tallies[round_record.initial_survivors]) $((post_round.initial_survivors))]")
            end
            margin = (second_to_last_tally - round_record.initial_tallies[reallocated_cand]) * margin_fac
            if margin < current_weight[1]
                result = pivot_record(round_num, margin, ballot_pile)
                if debug
                    println("   Voter $voter_num is pivotal in round $i: weight $current_weight[1], second to last $second_to_last_tally, last tally $(round_record.initial_tallies[reallocated_cand])")
                end
                return result
            elseif margin == current_weight[1]
                if reallocated_cand < second_to_last
                    result = pivot_record(round_num, margin * margin_fac, ballot_pile)
                    if debug
                        println("   Voter $voter_num is barely pivotal in round $i: weight $current_weight[1], second to last $second_to_last_tally, last tally $(round.initial_tallies[reallocated_cand])")
                    end
                    return result
                end
                
            end
        end
    end
    return result
end

function filter_pivot(pivot::Nothing, weight::Float64)
    return nothing
end

function filter_pivot(pivot::pivot_record, weight::Float64)
    pivot.margin *= 2 # this function is only called once we know they're only pivotal in one direction
    if pivot.margin <= weight # less than or equal technically could be wrong depending on candidate order, but so would less than. Not worth re-checking.
        return pivot
    end
    return nothing
end

function append_pivot!(pivots::Vector{pivot_record}, pivot::Nothing)
    return
end

function append_pivot!(pivots::Vector{pivot_record}, pivot::pivot_record)
    push!(pivots, pivot)
end

function find_equivalent_voters(voter_num::Int, election_glass::ElectionGlass, pre_pivot_record::STV_record, post_pivot_record::STV_record,
    winner_index::Int;
    weighted::Bool=true, quota::Float64=0., debug::Bool=false)
    """
    Given a voter number, an ElectionGlass, and a STV_record, give a vector of equivalencies to that voter, normalized to sum to 1.
    """
    model = election_glass.model

    n_candidates = count_candidates(model) * 2

    indexify_ = cand_id_to_index(n_candidates)
    if quota == 0.
        quota = get_quota(pre_pivot_record)
    end

    pre_pivots, post_pivots = find_pivotal_rounds(voter_num, election_glass,
        pre_pivot_record, post_pivot_record, winner_index, quota, indexify_, debug=debug)

    

    if length(pre_pivots) + length(post_pivots) == 0
        throw("Voter $voter_num is not pivotal in any round")
    end
    # Now we have the pivotal rounds and the weights in those rounds. For each other ballot, 
    # we can see if it agrees with the voter in those rounds. If so, then check its weight in those rounds.
    # If weighted is true, then weight by the weight in that round. 
    # Otherwise, weight is 1 iff its weight in that round >= the voter's weight then, 0 otherwise.

    if debug
        println("Pivotal rounds: $(pre_pivots) $(post_pivots)") 
    end
    equivalent_voters = zeros(Float64, length(election_glass.model.ballots))
    for i in 1:length(election_glass.model.ballots)
        if i == voter_num
            equivalent_voters[i] = 1.
            continue
        end
        ballot = get_ballot(model, i)

        result_cache = nothing # placeholder argument for check_ballot_repivotality
        is_pivot_weight = check_ballot_repivotality(ballot, pre_pivots, post_pivots, pre_pivot_record, post_pivot_record, winner_index, quota, indexify_, result_cache, debug=debug) 
        if is_pivot_weight == 0.
            continue
        end
        if weighted
            equivalent_voters[i] = is_pivot_weight
        else
            equivalent_voters[i] = 1.
        end

    end

    # Normalize
    if debug
        println("Equivalent voters sum: $(sum(equivalent_voters))")
    end
    equivalent_voters = equivalent_voters / sum(equivalent_voters)

    return equivalent_voters

end

function sample_one_BBVP(election::AbstractElection, n_winners::Int, winner_num::Int, get_winners::Function; 
    weighted::Bool=true, quota::Float64=0., debug::Bool=false, winners::Union{Vector{Int},Nothing}=nothing)
    """
    Given an election, the winner number to check, and a function to get winners,
    get a vector that's an unbiased estimate of the BBVP for each voter.
    """
    if winners == nothing
        winners = get_winners(election, n_winners)
    end
    winner_index = winners[winner_num]
    pivotal_voter, election_glass, record_lo, record_hi = sample_a_pivotal_voter_for(election, n_winners, winner_index, get_winners, InvertedPreferenceBallot, debug=debug)
    equivalent_voters = find_equivalent_voters(pivotal_voter, election_glass, record_lo, record_hi, winner_index, weighted=weighted, quota=quota, debug=debug)
    return equivalent_voters
end

function BBVP_multiestim_matrix(election::AbstractElection, n_winners::Int, winner_num::Int, get_winners::Function, num_estimates::Int; 
    weighted::Bool=true, quota::Float64=0., debug::Bool=false)
    """
    Get a matrix of BBVP estimates, one col per estimate, for a given winner.
    """
    winners = get_winners(election, n_winners)
    n_voters = length(election.ballots)
    BBVP_matrix = zeros(Float64, n_voters, num_estimates)
    for i in 1:num_estimates
        BBVP_matrix[:, i] = sample_one_BBVP(election, n_winners, winner_num, get_winners, weighted=weighted, quota=quota, debug=debug, winners=winners)
    end
    return BBVP_matrix
end

function BBVP_mean_var(election::AbstractElection, n_winners::Int, winner_num::Int, get_winners::Function, num_estimates::Int; 
    weighted::Bool=true, quota::Float64=0., debug::Bool=false)
    """
    Get the mean and variance of the BBVP estimates for a given winner.
    """
    BBVP_matrix = BBVP_multiestim_matrix(election, n_winners, winner_num, get_winners, num_estimates, weighted=weighted, quota=quota, debug=debug)
    return mean(BBVP_matrix, dims=2), var(BBVP_matrix, dims=2)
end

function test_pipe(random_seed::Int=0)
    Random.seed!(random_seed)
    DMM = VoterPower.DirichletSpatialGenerator(0.75, 0.05, 2., 2.)
    el = VoterPower.draw_electorate!(DMM, 50, 5)
    ex = VoterPower.get_election(el)
    w = VoterPower.STV_result(ex, 2)
    w1 = w[1]
    piv, gla, rec_lo, rec_hi = VoterPower.sample_a_pivotal_voter_for(ex, 2, w1, VoterPower.STV_result, InvertedPreferenceBallot, debug=true)


    # check that the voter is pivotal
    w_before = VoterPower.STV_result(gla, 2)
    gla.glass.how_deep += 1
    w_after = VoterPower.STV_result(gla, 2)
    gla.glass.how_deep -= 1
    println("Winners: $w")
    println("Pivotal voter: $piv")
    println("Mirror depth: $(gla.glass.how_deep)")
    println("Winners before: $w_before")
    println("Winners after: $w_after")

    eq = VoterPower.find_equivalent_voters(piv, gla, rec_lo, rec_hi, w_before[2])

    bbvp_mv = VoterPower.BBVP_mean_var(ex, 1, 1, VoterPower.STV_result, 100, debug=true)
    return (bbvp_mv, piv, eq, rec_lo, rec_hi, gla, ex, el, DMM)
end

end # module