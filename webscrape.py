from bs4 import BeautifulSoup
import requests
import time
import googlemaps
from datetime import date
from googleAPI import API_KEY #Needs a file with variable API_KEY. 
from csv import writer

start_time = time.time()
map_client = googlemaps.Client(API_KEY)
links = []
errors = 0
Ageerror = 0
success = 0

#Find link to every individual rental home
for i in range(1,215):
    url= "https://www.vuokraovi.com/vuokra-asunnot/helsinki?page=" + str(i) + "&pageType="
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    price = soup.find('span', class_="price").text
    address = soup.find('span', class_="address").text
    links += soup.find_all('a', class_="list-item-link")

links = list(set(links))
#Make and write to csv file
with open('Vuokraovi.csv', 'w', encoding= 'utf8', newline='') as f:
    thewriter = writer(f)
    header = ['Place', 'X-coord', 'Y-coord', 'Rooms', 'CommonR', 'Sauna', 'm^2', 'Age', 'Price']
    thewriter.writerow(header)

    #Visit every link one by one and pick the relevant information
    for link in links:
        url2 = "https://www.vuokraovi.com" + link['href']

        #Check if the link is actually a rental home and not an agency
        if url2.find('main') == -1:
            page = requests.get(url2)
            soup = BeautifulSoup(page.content, 'html.parser') 
            Place = 0

            #Find all the relevant information, if some are missing skip and mark error + 1
            try:
                Price = soup.find("th", string="Vuokra:").find_next_sibling("td").text
                Place = soup.find("span", itemprop="streetAddress").text
      
                Rooms = soup.find("th", string="Kuvaus:").find_next_sibling("td").text

                #Number of Common rooms are usually the first number. If something else than number discard the rental apartment
                CommonR = int(Rooms[0])

                #Sauna is marked as "+s" or "+S" in Rooms. Mark 1 if exist otherwise 0
                if Rooms.find('+s') != -1 or Rooms.find('+S') != -1 or Rooms.find(' s') != -1 or Rooms.find(' S') != -1 or Rooms.find(',s') != -1 or Rooms.find(',S') != -1:
                    Sauna = 1
                else:
                    Sauna = 0

                SqM = soup.find("th", string="Asuinpinta-ala:").find_next_sibling("td").text
                Age = soup.find("th", string="Rakennusvuosi:").find_next_sibling("td").text

                #Coordinates from google maps API
                Place_coord = map_client.geocode(Place + ' Helsinki')[0]['geometry']['location']
                Place_X = Place_coord['lat']
                Place_Y = Place_coord['lng']

                #Write the gathered info to csv file
                info = [Place, Place_X, Place_Y, Rooms, CommonR, Sauna, SqM, Age, Price]
                thewriter.writerow(info)
                success += 1

            except:
                errors += 1

#Stats to benchmark the code:
f = open("Stats.txt", "w+")
f.write("errors: " + str(errors) + "\n")
f.write("success: " + str(success) + "\n")
f.write("--- %s seconds ---" % (time.time() - start_time) + "\n")
f.write("Date: " + str(date.today()))