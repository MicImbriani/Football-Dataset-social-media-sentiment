# This file collects all the game data pertaining to each player from the football api
# Various files are produced depending on used fictures and game stats
# The destination is set to ../../../data/collected_with_some_processing/games, but can be changed


import os
import re
import datetime
import requests
import json
from unidecode import unidecode
from tqdm.auto import tqdm
import pandas as pd

from player_names_database import players, players_teams, players_seasons
from csv_column_names import columns_names, columns_raw_stats

headers = {
	"X-RapidAPI-Key": "insert-your-key-here",
	"X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
    }


def _get_teamID_from_name(teamName):
    # Helper function.
    # Given string of team name (e.g. "Manchester City"), returns team ID
    url = "https://api-football-v1.p.rapidapi.com/v3/teams"
    querystring = {"search": teamName}
    response = requests.request("GET", url, headers=headers, params=querystring)
    response = response.json()
    return int(response["response"][0]["team"]["id"])

def request_player(player_name, initial_team):
    # Given string of player name and team name, returns player ID and full name.
    rapidapi_url_endpoint = "https://api-football-v1.p.rapidapi.com/v3/"
    
    formatted_player_name = re.sub('[^0-9a-zA-Z ]+', '', player_name)
    parameters = {"team": _get_teamID_from_name(initial_team),
                  "search": unidecode(player_name).split()[-1]}

    response = requests.get(rapidapi_url_endpoint+"players",
                            headers=headers,
                            params=parameters)
    json_response = response.json()
    return str(json_response["response"][0]["player"]["id"]), \
           str(json_response["response"][0]["player"]["name"])
    

def get_teams(playerID):
    # Given a player ID, returns two dictionaries with the following key-value pairs:
    # season - team played
    # team - team ID
    url = "https://api-football-v1.p.rapidapi.com/v3/transfers"
    querystring = {"player": playerID}
    response = requests.request("GET", url, headers=headers, params=querystring)
    response = response.json()
    
    teamsPlayed = {}
    teamsIDs = {}
    transfers = response["response"][0]["transfers"]
    transferYears = list(map(lambda x: int(x["date"][:4]), transfers))
    for idx, transfer in enumerate(transfers):
        date = transfer["date"]
        teamOut, teamOutID = transfer["teams"]["out"]["name"], transfer["teams"]["out"]["id"]
        teamIn, teamInID = transfer["teams"]["in"]["name"], transfer["teams"]["in"]["id"]
        
        today = datetime.date.today()
        todayStr = str(today.year) + "-" + str(today.month) + "-" + str(today.day)
        
        todayMinus5 = datetime.datetime(today.year-5, today.month, today.day)
        todayStrMinus5 = str(today.year-5) + "-" + str(today.month) + "-" + str(today.day)
        
        d1 = datetime.datetime(int(date[:4]), int(date[5:7]), int(date[8:]))
        if d1 < todayMinus5:
            if len(teamsPlayed) == 0:
                teamsPlayed[teamIn] = todayStrMinus5 + "/" + todayStr
            else:
                previousTransferDate = transfers[idx-1]["date"]
                d2 = datetime.datetime(int(previousTransferDate[:4]), \
                     int(previousTransferDate[5:7]), \
                     int(previousTransferDate[8:]))
                if d2 < todayMinus5:
                    continue
                else:
                    teamsPlayed[teamIn] = todayStrMinus5 + "/" + previousTransferDate
            teamsIDs[teamIn] = teamInID
            if transferYears[idx] > max(transferYears):
                break
        else:
            if len(teamsPlayed) == 0 or teamsPlayed[list(teamsPlayed.keys())[0]][11:] != todayStr:
                teamsPlayed[teamIn] =  date + "/" + todayStr 
                teamsPlayed[teamOut] = todayStrMinus5 + "/" + date
            else:
                previousTransferDate = transfers[idx-1]["date"]
                teamsPlayed[teamIn] = date + "/" + str(previousTransferDate)
        
        teamsIDs[teamIn] = teamInID
        teamsIDs[teamOut] = teamOutID
            
    return teamsPlayed, teamsIDs

def get_seasons_played_from_player(player_name):
    # Given player name, returns his/her team information from our database.
    for p in players_seasons:
        name_wo_accent = unidecode(p)
        if (player_name in name_wo_accent):
            return players_seasons[p]  

def request_fixtures(player_name, seasonsPlayed, teamsPlayed, teamsIDs, showPrint=False):
    # Request and store locally the fixtures for each game in the seasons between 2022 (today) and 2017 (5 years ago).
    # The team and season year information are used to refine the search specifically to the player.
    # Then, create a list containing all the fixtures IDs, 
    # and a list with each fixture's date.
    fullListOfFixtures = []
    datesFixtures = []

    for season in seasonsPlayed:
        url = "https://api-football-v1.p.rapidapi.com/v3/fixtures"
        
        # get fixtures of team between dates when player played
        team = seasonsPlayed[season]
        fromDate, toDate = teamsPlayed[team][:10], teamsPlayed[team][11:]
        if showPrint:
            print("Season: ", season, "Search start date: ", fromDate, "Search end date: ", toDate, "Team: ", team)
        querystring = {"season": season, "team": teamsIDs[team], "from":fromDate,"to": toDate}
        response = requests.request("GET", url, headers=headers, params=querystring)
        
        # make json file for each team and season
        json_response = response.json()
        # saved on file
        temp_file_name = str(player_name).replace(" ", "_") + "_" + str(seasonsPlayed[season]).replace(" ", "_") + "_" + season + "_fixtures.json"
        with open(f'{player_name}/{temp_file_name}', 'wt') as temp_file_name:
            json.dump(json_response, temp_file_name)

        for fixture in json_response["response"]:
            fullListOfFixtures.append(fixture["fixture"]["id"])
            datesFixtures.append(fixture["fixture"]["date"][:10])

    return fullListOfFixtures, datesFixtures

def get_final_stats(player_name, full_player_name, fullListOfFixtures, datesFixtures):
    # Retrieves the player's per-game statistics of all the played games in the last 5 years.
    finalPlayerStats = []
    raw_finalPlayerStats = []

    urlFixturesPlayers = "https://api-football-v1.p.rapidapi.com/v3/fixtures/players"

    # for each game saved...
    print("Retrieving stats from games...")
    for currentFixture, fixtureDate in zip(tqdm(fullListOfFixtures), datesFixtures):
        # print()
        # print(currentFixture)
        # Get stats of ALL players
        querystring = {"fixture": str(currentFixture)}
        response = requests.request("GET", urlFixturesPlayers, headers=headers, params=querystring)
        
        # Check if fixture has game stats
        # if no stats -> skip game
        json_response = json.loads(response.text)
        if not len(json_response["response"]) > 0:
            continue
        else:
            player_stats = []
            currentTeam = ""
            # Else, check if our player played in that game; if yes, find his stats.
            for team in json_response["response"]:
                for player in team["players"]:
                    if player_name.split()[-1] in player["player"]["name"]:
                        player_stats.append(player["statistics"][0])
                        currentTeam = str(team["team"]["name"])
            
            # If player has any recorded stats, we store them both raw and refined.
            if len(player_stats)>0 and player_stats[0]["games"]["minutes"] is not None:
                player_stats = player_stats[0]
                opponent = [team for team in [json_response["response"][0]["team"]["name"], json_response["response"][1]["team"]["name"]] if team != currentTeam][0]
                date = fixtureDate
                # print(opponent)
                
                # Create entry for REFINED stats
                stats = [full_player_name,
                        opponent,
                        date,
                        player_stats["games"]["minutes"], 
                        player_stats["games"]["rating"], 
                        player_stats["shots"]["total"], 
                        player_stats["shots"]["on"], 
                        player_stats["goals"]["total"], 
                        player_stats["goals"]["assists"], 
                        player_stats["passes"]["total"], 
                        player_stats["tackles"]["total"], 
                        player_stats["tackles"]["blocks"], 
                        player_stats["duels"]["total"], 
                        player_stats["duels"]["won"],
                        player_stats["fouls"]["drawn"],
                        player_stats["penalty"]["scored"]]
                finalPlayerStats.append(stats)
                
                # Create entry for RAW stats
                raw_stats = [full_player_name, opponent,date]
                for statCategory in player_stats:
                    if player_stats[statCategory] is None:
                        raw_stats.append("NaN")
                    elif type(player_stats[statCategory]) is not int:
                        for stat in player_stats[statCategory]:
                            raw_stats.append(player_stats[statCategory][stat])
                    else:
                        raw_stats.append(player_stats[statCategory])
                raw_finalPlayerStats.append(raw_stats)
            # If player has NO recorded stats, just go on to next game.
            else:
                continue

    return finalPlayerStats, raw_finalPlayerStats

def export_to_csv(player_name, full_player_name, finalPlayerStats, raw_finalPlayerStats, destination_path):
    # Exports the data to two .csv files. One for raw stats, one for refined stats.
    df = pd.DataFrame(data=finalPlayerStats, columns=columns_names)
    df.to_csv(os.path.join(destination_path, f"{full_player_name}.csv"), na_rep="None")

    df_raw = pd.DataFrame(data=raw_finalPlayerStats, columns=columns_raw_stats)
    df_raw.to_csv(os.path.join(destination_path, f"{full_player_name}RAW.csv"), na_rep="None")

def get_data_from_player(player_name, destination_dir, showPrint=False):
    # Functions for executing the player data collection pipeline.
    team_name = ""
    if not(player_name in players):
        raise RuntimeError("Error: Given player is not in our top players list")
    else:
        team_name = players_teams[players.index(player_name)] 

    playerID, full_player_name = request_player(player_name, team_name)
    teamsPlayed, teamsIDs = get_teams(playerID)
    seasonsPlayed = get_seasons_played_from_player(player_name)

    # Create new folder to insert player's data.
    destination_path = os.path.join(destination_dir, player_name)
    try:
        os.makedirs(destination_path)
    except: 
        pass

    fixtures, fixtureDates = request_fixtures(player_name, seasonsPlayed, teamsPlayed, teamsIDs, showPrint)
    final_stats, raw_final_stats = get_final_stats(player_name, full_player_name, fixtures, fixtureDates)

    export_to_csv(player_name, full_player_name, final_stats, raw_final_stats, destination_path)




############### main ###############

if __name__ == '__main__':
    destination_dir = '../../../data/collected_with_some_processing/games'
    print("Insert player name (first name + last name): ")
    print("(type 'TOP10' to generate data for all top 10 players in our database)")
    input_name = input()
    if input_name == "TOP10":
        for player in tqdm(players):
            print()
            print(player)
            get_data_from_player(player, destination_dir)
    else:
        get_data_from_player(input_name, destination_dir, showPrint=True)

############### main END ###############
