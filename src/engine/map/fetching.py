import time
import requests

OVERPASS_URL = "http://overpass-api.de/api/interpreter"


def fetch_road_data(city_name):
    """
    Fetches the road data for the specified city from the OpenStreetMap database.
    """
    # Requête Overpass pour fetch les routes (highways), données non triées
    overpass_query = f"""
    [out:json];
    area[name="{city_name}"]["boundary"="administrative"]["admin_level"="8"]->.searchArea;
    (
        way["highway"="primary"](area.searchArea);
        way["highway"="secondary"](area.searchArea);
        way["highway"="tertiary"](area.searchArea);
        way["highway"="residential"](area.searchArea);
    );
    out body;
    >;
    out skel qt;
    """

    print("Fetching data from Overpass API...")
    start = time.time()
    response = requests.get(OVERPASS_URL, params={'data': overpass_query})
    data = response.json()
    print(f"Time fetching: ({time.time() - start:.3f}s)")

    return data