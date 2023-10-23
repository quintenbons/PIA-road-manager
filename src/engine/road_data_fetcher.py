import requests
import json

def fetch_road_data(city_name):
    # Define the Overpass query. This query fetches highways in the specified city.
    overpass_query = f"""
    [out:json];
    area[name="{city_name}"];
    (way[highway](area);
     way[railway](area);
    );
    out body;
    >;
    out skel qt;
    """
    
    # Define the Overpass API URL
    overpass_url = "http://overpass-api.de/api/interpreter"
    
    # Send the request to the Overpass API and get the JSON response
    print("Sending the request to the Overpass API...")
    response = requests.get(overpass_url, params={'data': overpass_query})
    data = response.json()
    
    # Save the JSON data to a file for easier inspection and debugging
    print("Saving the JSON data to a file...")
    with open('road_data.json', 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    
    return data

# Call the function with the name of your city
road_data = fetch_road_data("Saint-Trinit")
