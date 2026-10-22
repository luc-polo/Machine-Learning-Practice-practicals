

from numpy import sort
import numpy as np

def find_arbitrage(odds):
    max_odds = np.zeros((N,K))
    max_book = np.zeros((N,K))
    arbitrage = []
    for i, match in enumerate(odds):
        for j in range(K):
            max_odds[i][j]= max(match[:,j])
            max_book[i][j] = np.argmax(match[:, j])

        if sum(1/max_odds[i]) < 1:
            arbitrage.append(True)
        else:
            arbitrage.append(False)

    return arbitrage, max_odds, max_book


def find_survivor(n):
    list= [i for i in range(n)]
    while len(list)>1:
        old_list = list.copy()
        list.pop(0)
        list.pop(0)
        list.append(old_list[0])
    return list[0] +1


def detect_groups(bets, max_delay, min_co_bets, min_group_size):
    groups = {}
    matches_issues = {}
    for row in bets:
        if (row[1], row[2]) not in matches_issues:
            matches_issues[(row[1], row[2])] = []
        matches_issues[(row[1], row[2])].append([row[0], row[3]])


    for match in matches_issues.values():

        for i in range(len(match)):
            for j in range(i+1, len(match)):
                bet1=match[i]
                bet2=match[j]

                if abs(bet2[1]-bet1[1])>max_delay:
                    pass
                else:
                    pair= [bet1[0], bet2[0]]
                    if pair[0]!= pair[1]:
                        pair.sort()
                        pair = tuple(pair)
                        groups[pair] = groups.get(pair,0) +1

    new_groups = groups.copy()
    for key, value in groups.items():
        if value<min_co_bets:
            del new_groups[key]
    groups = list(new_groups.keys())


    change = True
    while change == True:
        change = False
        for i in range(len(groups)):
            if change:
                break
            for j in range(i+1,len(groups)):
                if change:
                    break
                for player in groups[j]:
                    if player in groups[i]:
                        groups[i] = tuple(set(groups[j]+ groups[i]))
                        del groups[j]
                        change = True
                        break

    groups = [group for group in groups if len(group)>= min_group_size]
    
    return groups

