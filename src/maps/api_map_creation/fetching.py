import requests
from .utils import timing

OVERPASS_URL = "http://overpass-api.de/api/interpreter"

@timing
def fetch_road_data(city_name):
    """
    Fetches the road data for the specified city from the OpenStreetMap database.
    """
    # Requête Overpass pour fetch les routes (highways), données non triées
    overpass_query = f"""
    [out:json];
    area[name="{city_name}"]["boundary"="administrative"]["admin_level"="8"]->.searchArea;
    (
        way["highway"="trunk"](area.searchArea);
        way["highway"="primary"](area.searchArea);
        way["highway"="secondary"](area.searchArea);
        way["highway"="tertiary"](area.searchArea);
        way["highway"="unclassified"](area.searchArea);
        way["highway"="residential"](area.searchArea);
        way["highway"="service"](area.searchArea);
    );
    out body;
    >;
    out skel qt;
    """

    print("Fetching data from Overpass API...")
    response = requests.get(OVERPASS_URL, params={'data': overpass_query})
    data = response.json()

    return data
