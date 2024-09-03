module VoterPower

using Random
using Maybe

abstract type AbstractBallot end

struct PreferenceBallot <: AbstractBallot
    preferences::Vector{Int16} # ordered list of candidates. Should be a permutation of 1:n_candidates
end

struct InvertedPreferenceBallot <: AbstractBallot
    model::PreferenceBallot
end

abstract type AbstractElection end

struct Election <: AbstractElection
    n_candidates::Int
    ballots::Vector{AbstractBallot}
end

struct ElectionWithMirrors <: AbstractElection
    model::Election
    inversions::Vector{Bool}
end

struct STV_round_record
    initial_survivors::BitVector #Technically this is redundant, but it avoids recalculating.
    initial_tallies::Vector{Float64} #copy of the weighted_tallies at the beginning of the round
    winner_found::Bool #whether a winner was found in this round
    reallocated_cand::Int #the id of the candidate who was reallocated in this round
end

struct STV_record 
    n_winners::Int
    n_voters::Int
    n_candidates::Int
    rounds::Vector{STV_round_record}
    winners::Vector{Int}
end

function STV_result(election::Election, n_winners::Int;
        assert_invariants::Bool=false,
        record::Union{STV_record, Nothing}=nothing)
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
    n_voters = length(election.ballots)
    n_candidates = election.n_candidates * 2 # Each candidate has a mirror anti-candidate. 
        # The anti-candidate is the candidate with the same index, but with a negative sign.
        # If you're running a normal election, the anti-candidates will never win, so you can ignore them.

    # Create and initialize running variables
    surviving_candidates = trues(n_candidates)
    winners = zeros(Int, n_winners) # Zero is not a valid candidate index, even though negative indices are valid anti-candidates
    winners_so_far = 0
    round = 1
    ballot_piles = zeros(Int, [n_candidates, n_voters])
    ballot_pile_sizes = zeros(Int, n_candidates)
    weighted_tallies = zeros(Float64, n_candidates)
    ballot_weights = ones(Float64, n_voters)

    # Call the mutating version of the function
    return STV_result_inner!(election, n_winners, surviving_candidates, winners, 
            winners_so_far, round, ballot_piles, ballot_pile_sizes, weighted_tallies, ballot_weights;
            assert_invariants=assert_invariants, record=record)
end

function STV_result_inner!(election::Election, 
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
                record::Union{STV_record, Nothing}=nothing)
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
    
    # record global params
    if record != nothing
        record.n_winners = n_winners
        record.n_voters = length(election.ballots)
        record.n_candidates = election.n_candidates
        record.rounds = []
        record.winners = winners
    end


    # Calculate the Droop quota
    n_voters = length(election.ballots)
    quota = n_voters / (n_winners + 1) # technically this isn't the Droop quota because we're not adding epsilon, but for the invariants above this is actually better.
    n_candidates = election.n_candidates * 2 # Each candidate has a mirror anti-candidate. 
        # The anti-candidate is the candidate with the same index, but with a negative sign.
        # If you're running a normal election, the anti-candidates will never win, so you can ignore them.
    deindexify = cand_index_to_id(n_candidates) # Convert from candidate index (1 to n_candidates) to candidate id (-n_candidates/2 to n_candidates/2)
        # candidate index has the anti-candidates after the candidates, in the same order.
    indexify = cand_id_to_index(n_candidates) # Convert from candidate id (-n_candidates/2 to n_candidates/2) to candidate index (1 to n_candidates)

    do_initial_tallies!(election, n_winners, surviving_candidates, winners, winners_so_far, round, ballot_piles, ballot_pile_sizes, weighted_tallies, ballot_weights)
    # keep doing rounds until we have enough winners
    while winners_so_far < n_winners

        # Precondition assertion
        @assert round <= n_candidates

        # Record the round
        if record != nothing
            push!(record.rounds, STV_round_record(copy(surviving_candidates), copy(weighted_tallies), false, 0))
            # We'll set winner_found and reallocated_cand later
        end

        # Check if any candidate has more votes than the quota
        if max(weighted_tallies[surviving_candidates]) >= quota
                # Note `>=` ... this means we won't have a tiebreaker if the last two candidates have exactly the quota. 

            # If so, elect the candidate with the most votes
            next_survivor = argmax(weighted_tallies[surviving_candidates]) # This gets the index of the winner *among the surviving candidates*.
                # This is not the same as the candidate id or even the candidate index.
            next_winner_index = get_survivor_index(next_survivor, surviving_candidates, n_candidates)
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
            next_winner = deindexify(next_winner_index)

            # Record the round
            if record != nothing
                record.rounds[end].winner_found = true
                record.rounds[end].reallocated_cand = next_winner
            end

            winners[winners_so_far + 1] = next_winner
            winners_so_far += 1
            surviving_candidates[next_winner_index] = false
            # Transfer the surplus votes
            surplus = weighted_tallies[next_winner_index] - quota
            weight_ratio = surplus / weighted_tallies[next_winner_index]
            for i in 1:ballot_pile_sizes[next_winner_index]
                voter_num = ballot_piles[next_winner_index, i]
                ballot = get_ballot(election, voter_num)
                new_pile = best_surviving_preference(ballot, surviving_candidates, n_candidates, indexify)
                if new_pile == 0
                    ballot_weights[voter_num] = 0.
                else
                    ballot_piles[new_pile, ballot_pile_sizes[new_pile] + 1] = voter_num
                    ballot_pile_sizes[new_pile] += 1
                    ballot_weights[voter_num] *= weight_ratio
                end
            end
            break
        end

        
        # No candidate has more votes than the quota, so eliminate the candidate with the fewest votes
        losing_survivor = argmin(weighted_tallies[surviving_candidates])
        loser_index = get_survivor_index(losing_survivor, surviving_candidates, n_candidates)
        loser = deindexify(loser_index)

        # Record the round
        if record != nothing
            record.rounds[end].reallocated_cand = loser
            # winner_found is already false
        end

        surviving_candidates[loser_index] = false
        for i in 1:ballot_pile_sizes[loser_index]
            voter_num = ballot_piles[loser_index, i]
            ballot = get_ballot(election, voter_num)
            new_pile = best_surviving_preference(ballot, surviving_candidates, n_candidates, indexify)
            if new_pile == 0
                ballot_weights[voter_num] = 0.
            else
                ballot_piles[new_pile, ballot_pile_sizes[new_pile] + 1] = voter_num
                ballot_pile_sizes[new_pile] += 1
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

    end

    #record winners
    if record != nothing
        record.winners = winners
    end

    return winners
end

# Helper functions

function do_initial_tallies!(election::Election, 
                n_winners::Int, 
                surviving_candidates::BitVector, 
                winners::Vector{Int}, 
                winners_so_far::Int, 
                round::Int, 
                ballot_piles::Matrix{Int}, 
                ballot_pile_sizes::Vector{Int}, 
                weighted_tallies::Vector{Float64}, 
                ballot_weights::Vector{Float64})
    n_voters = length(election.ballots)
    n_candidates = election.n_candidates * 2 # Each candidate has a mirror anti-candidate. 
        # The anti-candidate is the candidate with the same index, but with a negative sign.
        # If you're running a normal election, the anti-candidates will never win, so you can ignore them.
    indexify = cand_id_to_index(n_candidates) # Convert from candidate id (-n_candidates/2 to n_candidates/2) to candidate index (1 to n_candidates)

    # Initialize the ballot piles
    for i in 1:n_candidates
        ballot_pile_sizes[i] = 0
    end
    for i in 1:n_voters
        ballot = get_ballot(election, i)
        pile = best_surviving_preference(ballot, surviving_candidates, n_candidates, indexify)
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

function get_ballot(election::ElectionWithMirrors, voter_num::Int)
    if election.inversions[voter_num]
        return InvertedPreferenceBallot(election.model.ballots[voter_num])
    else
        return election.model.ballots[voter_num]
    end
end

function best_surviving_preference(ballot::PreferenceBallot, surviving_candidates::BitVector, n_candidates::Int, indexify::Function;
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

function best_surviving_preference(ballot::InvertedPreferenceBallot, surviving_candidates::BitVector, n_candidates::Int, indexify::Function)
    return best_surviving_preference(ballot.model, surviving_candidates, n_candidates, indexify; dont_invert_ballot=false)
end

function cand_id_to_index(n_candidates::Int)
    return function(cand_id::Int)
        if cand_id > 0
            return cand_id
        else
            return n_candidates - cand_id
        end
    end
end

function cand_index_to_id(n_real_candidates::Int)
    return function(cand_index::Int)
        if cand_index <= n_real_candidates
            return cand_index
        else
            return n_candidates - cand_index
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
                record::Union{STV_record, Nothing}=nothing)
    reinitialize_running_variables!(election, n_winners, surviving_candidates, winners, winners_so_far, round, ballot_piles, ballot_pile_sizes, weighted_tallies, ballot_weights)
    return STV_result_inner!(election, n_winners, surviving_candidates, winners, winners_so_far, 
            round, ballot_piles, ballot_pile_sizes, weighted_tallies, ballot_weights;
            assert_invariants=assert_invariants, record=record)
end

# Functions to create random voters and elections

function random_preference_ballot(n_candidates::Int)
    return PreferenceBallot(randperm(n_candidates))
end

function random_STV_election(n_candidates::Int, n_voters::Int)
    ballots = [PreferenceBallot(randperm(n_candidates)) for i in 1:n_voters]
    return Election(n_candidates, ballots)
end

end