Title: Web Scraping Tutorial
Date: 2018-04-10 11:00am
Category: Web Scraping
Tags: tutorial, web scraping, BeautifulSoup, Politifact.com, Trump, fact checking
Slug: scraping-tutorial-post
Author: William Miller
Summary: Scraping PolitiFact.com for fact check data.

In preparation for a project comparing President Trump's veracity to that of contemporary U.S politicians, I have scraped some data from www.PolitiFact.com. Since this is a fairly basic data scraping project, I thought I would take a bit to make this into a tutorial on the subject. My goal here is to write some code that will:
<ol><li> Take the name of any politician on PolitiFact.com in a list</li>
<li>Retrieve the data from Politifact for: the date they made a statement, the text of that statement, the Politifact rating of that statement</li>
<li>Store the resulting data in a dataframe and export to a CSV file</li>
<ol>

#### Libraries
First, there are several libraries that I need to import for just about any data scraping project. <ul>
<li>"Beautiful Soup" makes navigating HTML much more convenient and is nearly indispensible for this purpose.</li>
<li>We will be using the "get" function from the "requests" library to retrieve the HTML from specific URLs.
<li>It is highly likely in any web scraping project is going to require searching or processing text using regular expressions, so importing the "re" library is also necessary.</li>
<li>Any time you are requesting lot of data from a website, it is necessary to space out the requests you are making in order to avoid looking like you're instigating a DDoS attack, getting your IP banned temporarily. We will import the sleep function from the "time" library and "randint" from "random" to let us wait for random time intervals between requests.</li>
<li>As with basically any data project, it is very likely, if not definite, that we will need functions from the "pandas" and "numpy" libraries. We will go ahead and import both.</li>
<li>While this is specific to this particular project, I know that I will be parsing some dates as a part of this, so I will go ahead and load the "parser" function from the "dateutil" library. </li></ul>



```python
from bs4 import BeautifulSoup
import requests
import re

from time import sleep
from random import randint

import numpy as np
import pandas as pd

from dateutil import parser
```

#### Function to show progress
I mentioned above that we will be retrieving a lot of data from a specific website, waiting random amounts of time between requests. This means that there will be a fair amount of waiting time involved in this progress as we step through lists of URLs to retrieve our data. Because of this, I'm going to set up a quick progress-tracking function. It simply takes a list and an element in that list as input, and prints the number of the element entered compared to the whole. For instance if I passed in "c" and "[a,b,c,d,e], this function will print "step 3 of 5". This give a rough idea of how long the scraping process will take and lets us know it's doing something.


```python
def show_progress(part, whole):
    """
    Inputs:
    part = element of list
    whole = list
    ---------
    Function:
    Print the number of element "part" is within "whole"
    ---------
    Output:
    None
    """
    path = len(whole)
    step = whole.index(part) + 1
    progress = "step " + str(step) + " of " + str(path)
    print(progress)
```

#### Recognize and exploit patterns in URLs
At this point, I had to go through a process I cannot really show here. I went to Politifact website and viewed which contemporary politicians had fact checking data on their website, and I selected a few of these. I then investigated how to get total number of fact checks for each person, and looked for patterns in the URLs I would need to request.

For instance, I noticed if I clicked on "personalities", then "Donald Trump", and then selected "See all" for statements by Trump, it returned 28 pages of fact checks. I noticed that clicking through these pages returned different URLs, so page 3 had a URL of "http://www.politifact.com/personalities/donald-trump/statements/?page=3&list=speaker". I ensured that I could plug in any page number to the right of "page=", and it would go to that page.

Looking at other politicians, I noticed that their fact-check pages followed the same convention regarding their names, where Donald Trump's page contained "personalities/donald-trump", Barack Obama's page contained "personalities/barack-obama". "Firstname Lastname" was consistently formatted as "firstname-lastname". We can combine this information with the above to return a complete list of fact-checks for any politician listed on PolitiFact.

#### Format politican names for URLs, set up URL lookup data organization
To use this info, I will make a list of the names of the politician's I want ratings from. My aim is to do this in such a way that I could add any name of any person with fact checks on PolitiFact, and it will retrieve the data on them. I will then write some code to format that list in accordance with that URL name convention I mentioned.

One thing that can be immensely useful in any data scraping project is to set up a dictionary to store information in based on what you're looking up. I will therefore create the dictionary "person_lookup_dict" with each politican's name as the key. I will then initialize a dictionary stored under each of their names that I can the url-formatted names to. Later, I will add additional lookup-related information to this dictionary.

While it can be somewhat trickier at times to add and retrieve information from a dictionary rather than a bunch of separate lists, this will help immensely to keep all of the information we need to use retrieve the correct URLs organized. It will also help ensure that I can add names to the list of people whose fact-check data I wish to retrieve without modifying my code.


```python
# People to retrieve fact check data for from PolitiFact.com.
person_list = ['Donald Trump', 'Barack Obama', 'Mike Pence', 'Paul Ryan',
               'Nancy Pelosi', 'Mitch McConnell', 'Charles Schumer']

# Initialize a dictionary with the names above as keys.
person_lookup_dict = dict.fromkeys(person_list, {})

# Initialize a dictionary under each of these keys containing the URL formatting of each name.
for person in person_lookup_dict:
    person_lookup = person.lower()
    person_lookup = person_lookup.replace(" ", "-")

    person_lookup_dict[person] = {'urlname':person_lookup}

# Show the result.
person_lookup_dict
```




    {'Barack Obama': {'urlname': 'barack-obama'},
     'Charles Schumer': {'urlname': 'charles-schumer'},
     'Donald Trump': {'urlname': 'donald-trump'},
     'Mike Pence': {'urlname': 'mike-pence'},
     'Mitch McConnell': {'urlname': 'mitch-mcconnell'},
     'Nancy Pelosi': {'urlname': 'nancy-pelosi'},
     'Paul Ryan': {'urlname': 'paul-ryan'}}



#### Retrieve number of pages per person for URL lookup
At this point, we need to delve into the HTML data on page 1 for at least a couple of the people we're retrieving data on, with the aim of writing code to scrape the total number of pages of fact-checks each person has. This will prevent us from having to manually enter this number for each person, and manually update that number any time we wish to run this code again that the number of pages may have changed.

I start by pulling the HTML source from page 1 for each person. Looking at one of these webpages in my browser, at the bottom of the screen I can see the text "Page 1 of ??" on the bottom of it, and it's a safe bet that is encoded in the HTML and can be retrieved using BeautifulSoup. Looking at the HTML data from page directly (it's easiest to visit the URL in a browser, right-click, and then hit "View Page Source"), and searching for that text, I find that it is contained in a tag "step-links\_\_current". Searching for this, I can find that there are two instance of this per page. Checking other pages, I find that this is consistently the case.

The plan therefore becomes the following:
<ol>
<li>Request the data for page 1 for each person in our list. (Waiting between each request.)</li>
<li>Find the string "Page 1 of ??"</li>
<li>Process that string to get only the integer for the maximum number of pages.</li>
<li>Store that value for each person in the dictionary, in the dictionary under each name.</li>
</ol>


```python
for person in person_lookup_dict:
    # Request data from page 1 for each person in list.
    person_url=person_lookup_dict[person]['urlname']
    start_page = requests.get("http://www.politifact.com/personalities/" + person_url + "/statements/?page=1&list=speaker")
    start_soup = BeautifulSoup(start_page.text, 'html.parser')

    # Wait a random amount of time between 10 and 20 seconds.
    #If an error is returned, state the status code and break the loop.
    sleep(randint(10,20))
    if start_page.status_code != 200:
        print('We may have a problem here.', start_page.status_code)
        break

    # Find the string "Page 1 of ??" (contained within tags "class_=...").
    # Process down to integer value of max pages.
    num_page_str = start_soup.find(class_="step-links__current").find_next(class_="step-links__current")
    num_page_sub_str = re.search( r'(\d+) of (\d+)', str(num_page_str), re.M)
    person_lookup_dict[person]['urlpages']= int(num_page_sub_str.group(2))

    # Show progress.
    show_progress(person, list(person_lookup_dict.keys()))    

# Show results.
person_lookup_dict
```

    step 1 of 7
    step 2 of 7
    step 3 of 7
    step 4 of 7
    step 5 of 7
    step 6 of 7
    step 7 of 7





    {'Barack Obama': {'urlname': 'barack-obama', 'urlpages': 31},
     'Charles Schumer': {'urlname': 'charles-schumer', 'urlpages': 1},
     'Donald Trump': {'urlname': 'donald-trump', 'urlpages': 28},
     'Mike Pence': {'urlname': 'mike-pence', 'urlpages': 3},
     'Mitch McConnell': {'urlname': 'mitch-mcconnell', 'urlpages': 2},
     'Nancy Pelosi': {'urlname': 'nancy-pelosi', 'urlpages': 2},
     'Paul Ryan': {'urlname': 'paul-ryan', 'urlpages': 5}}



#### Generate URL list
We now have everything we need to generate a list of URLs for each person in the list. We can now iterate through each key (politician name) in the dictionary, retrieve the URL-formatted name and the number of pages for each person, and then generate a list of URLs from that info. I will store the URLs in the same person_lookup_dict that the other data is stored in, just to keep it all consistent and neat.


```python
# For each key in person_lookup_dict, retrieve that data to build correct URLs, then build URLs.
for person in person_lookup_dict:
    person_lookup_dict[person]['urllist'] = []
    person_name_url = person_lookup_dict[person]['urlname']
    for i in range(1, person_lookup_dict[person]['urlpages'] + 1):
        url = "http://www.politifact.com/personalities/" + person_name_url\
        + "/statements/?page="+ str(i) +"&list=speaker"

        # Store URLs in person_lookup_dict.
        person_lookup_dict[person]['urllist'].append(url)

#Show results.
person_lookup_dict
```




    {'Barack Obama': {'urllist': ['http://www.politifact.com/personalities/barack-obama/statements/?page=1&list=speaker',
       'http://www.politifact.com/personalities/barack-obama/statements/?page=2&list=speaker',
       'http://www.politifact.com/personalities/barack-obama/statements/?page=3&list=speaker',
       'http://www.politifact.com/personalities/barack-obama/statements/?page=4&list=speaker',
       'http://www.politifact.com/personalities/barack-obama/statements/?page=5&list=speaker',
       'http://www.politifact.com/personalities/barack-obama/statements/?page=6&list=speaker',
       'http://www.politifact.com/personalities/barack-obama/statements/?page=7&list=speaker',
       'http://www.politifact.com/personalities/barack-obama/statements/?page=8&list=speaker',
       'http://www.politifact.com/personalities/barack-obama/statements/?page=9&list=speaker',
       'http://www.politifact.com/personalities/barack-obama/statements/?page=10&list=speaker',
       'http://www.politifact.com/personalities/barack-obama/statements/?page=11&list=speaker',
       'http://www.politifact.com/personalities/barack-obama/statements/?page=12&list=speaker',
       'http://www.politifact.com/personalities/barack-obama/statements/?page=13&list=speaker',
       'http://www.politifact.com/personalities/barack-obama/statements/?page=14&list=speaker',
       'http://www.politifact.com/personalities/barack-obama/statements/?page=15&list=speaker',
       'http://www.politifact.com/personalities/barack-obama/statements/?page=16&list=speaker',
       'http://www.politifact.com/personalities/barack-obama/statements/?page=17&list=speaker',
       'http://www.politifact.com/personalities/barack-obama/statements/?page=18&list=speaker',
       'http://www.politifact.com/personalities/barack-obama/statements/?page=19&list=speaker',
       'http://www.politifact.com/personalities/barack-obama/statements/?page=20&list=speaker',
       'http://www.politifact.com/personalities/barack-obama/statements/?page=21&list=speaker',
       'http://www.politifact.com/personalities/barack-obama/statements/?page=22&list=speaker',
       'http://www.politifact.com/personalities/barack-obama/statements/?page=23&list=speaker',
       'http://www.politifact.com/personalities/barack-obama/statements/?page=24&list=speaker',
       'http://www.politifact.com/personalities/barack-obama/statements/?page=25&list=speaker',
       'http://www.politifact.com/personalities/barack-obama/statements/?page=26&list=speaker',
       'http://www.politifact.com/personalities/barack-obama/statements/?page=27&list=speaker',
       'http://www.politifact.com/personalities/barack-obama/statements/?page=28&list=speaker',
       'http://www.politifact.com/personalities/barack-obama/statements/?page=29&list=speaker',
       'http://www.politifact.com/personalities/barack-obama/statements/?page=30&list=speaker',
       'http://www.politifact.com/personalities/barack-obama/statements/?page=31&list=speaker'],
      'urlname': 'barack-obama',
      'urlpages': 31},
     'Charles Schumer': {'urllist': ['http://www.politifact.com/personalities/charles-schumer/statements/?page=1&list=speaker'],
      'urlname': 'charles-schumer',
      'urlpages': 1},
     'Donald Trump': {'urllist': ['http://www.politifact.com/personalities/donald-trump/statements/?page=1&list=speaker',
       'http://www.politifact.com/personalities/donald-trump/statements/?page=2&list=speaker',
       'http://www.politifact.com/personalities/donald-trump/statements/?page=3&list=speaker',
       'http://www.politifact.com/personalities/donald-trump/statements/?page=4&list=speaker',
       'http://www.politifact.com/personalities/donald-trump/statements/?page=5&list=speaker',
       'http://www.politifact.com/personalities/donald-trump/statements/?page=6&list=speaker',
       'http://www.politifact.com/personalities/donald-trump/statements/?page=7&list=speaker',
       'http://www.politifact.com/personalities/donald-trump/statements/?page=8&list=speaker',
       'http://www.politifact.com/personalities/donald-trump/statements/?page=9&list=speaker',
       'http://www.politifact.com/personalities/donald-trump/statements/?page=10&list=speaker',
       'http://www.politifact.com/personalities/donald-trump/statements/?page=11&list=speaker',
       'http://www.politifact.com/personalities/donald-trump/statements/?page=12&list=speaker',
       'http://www.politifact.com/personalities/donald-trump/statements/?page=13&list=speaker',
       'http://www.politifact.com/personalities/donald-trump/statements/?page=14&list=speaker',
       'http://www.politifact.com/personalities/donald-trump/statements/?page=15&list=speaker',
       'http://www.politifact.com/personalities/donald-trump/statements/?page=16&list=speaker',
       'http://www.politifact.com/personalities/donald-trump/statements/?page=17&list=speaker',
       'http://www.politifact.com/personalities/donald-trump/statements/?page=18&list=speaker',
       'http://www.politifact.com/personalities/donald-trump/statements/?page=19&list=speaker',
       'http://www.politifact.com/personalities/donald-trump/statements/?page=20&list=speaker',
       'http://www.politifact.com/personalities/donald-trump/statements/?page=21&list=speaker',
       'http://www.politifact.com/personalities/donald-trump/statements/?page=22&list=speaker',
       'http://www.politifact.com/personalities/donald-trump/statements/?page=23&list=speaker',
       'http://www.politifact.com/personalities/donald-trump/statements/?page=24&list=speaker',
       'http://www.politifact.com/personalities/donald-trump/statements/?page=25&list=speaker',
       'http://www.politifact.com/personalities/donald-trump/statements/?page=26&list=speaker',
       'http://www.politifact.com/personalities/donald-trump/statements/?page=27&list=speaker',
       'http://www.politifact.com/personalities/donald-trump/statements/?page=28&list=speaker'],
      'urlname': 'donald-trump',
      'urlpages': 28},
     'Mike Pence': {'urllist': ['http://www.politifact.com/personalities/mike-pence/statements/?page=1&list=speaker',
       'http://www.politifact.com/personalities/mike-pence/statements/?page=2&list=speaker',
       'http://www.politifact.com/personalities/mike-pence/statements/?page=3&list=speaker'],
      'urlname': 'mike-pence',
      'urlpages': 3},
     'Mitch McConnell': {'urllist': ['http://www.politifact.com/personalities/mitch-mcconnell/statements/?page=1&list=speaker',
       'http://www.politifact.com/personalities/mitch-mcconnell/statements/?page=2&list=speaker'],
      'urlname': 'mitch-mcconnell',
      'urlpages': 2},
     'Nancy Pelosi': {'urllist': ['http://www.politifact.com/personalities/nancy-pelosi/statements/?page=1&list=speaker',
       'http://www.politifact.com/personalities/nancy-pelosi/statements/?page=2&list=speaker'],
      'urlname': 'nancy-pelosi',
      'urlpages': 2},
     'Paul Ryan': {'urllist': ['http://www.politifact.com/personalities/paul-ryan/statements/?page=1&list=speaker',
       'http://www.politifact.com/personalities/paul-ryan/statements/?page=2&list=speaker',
       'http://www.politifact.com/personalities/paul-ryan/statements/?page=3&list=speaker',
       'http://www.politifact.com/personalities/paul-ryan/statements/?page=4&list=speaker',
       'http://www.politifact.com/personalities/paul-ryan/statements/?page=5&list=speaker'],
      'urlname': 'paul-ryan',
      'urlpages': 5}}



#### Retrieve and parse data
Here is the real substance to what we are doing. For the sake of organization, I created a function that parses the data I want to retrieve from the HTML I requested using the generated URLs. This part of the process involves more HTML investigation like that done for finding the page number data above, except here I'm looking for tags that allow be to locate the date a statement was made, the text of that statement, and the truth-rating assigned to it.

I found that the tag "class\_='statement\_\_source'" containing the politician's name (the version in the list prior to URL formatting) contained all the data I was look for. Below, I put the HTML containing that data into a list with each statement as an element. I then pass that list to a function that steps through it, scrapes the data I want, and appends only that data to a dictionary (truth_data).


```python
# See code below before parsing function code.
def truth_extractor(person, fact_checks, truth_data):
    """
    Inputs:
    person = Name of person making the statement in question.

    fact-checks = list of HTML elements containing that fact-check data to be scraped

    truth_data = dictionary of data scraped so far to be appended and returned.
    ---------
    Function:
    Step through list of HTML elements containing desired data, locate data, parse into desired format,
    append to dictionary.
    ---------
    Output:
    truth_data = dictionary with scraped data appended
    """
    # Iterate over items stored in fact_checks list.
    for check in fact_checks:
        #Within this item, located the tag "class_='statement'.
        statement = check.find_parent(class_='statement')

        #Locate the the data for statement date, truth rating (under meter), and text using associated tags.
        statement_date = statement.find_all("span", class_="article__meta")
        statement_meter = statement.find_all(class_="meter")
        statement_text = statement.find_all(class_="statement__text")

        #Perform first parsing of each string retrieved above. Text needs no parsing.
        parse_date_1 = statement_date[0].text
        parse_meter_1 = re.findall( r'(\"(.+?)\")', str(statement_meter[0]), re.M)
        parse_text_final = statement_text[0].text

        #Perform further parsing of date, final parsing of truth rating string.
        parse_date_2 = parse_date_1.replace("on ", "")
        parse_meter_final = str(parse_meter_1[2][0].replace('"', ''))

        #Perform final parsing of date
        parse_date_final = parser.parse(parse_date_2).date()

        #Append scraped data to dictionary initialized below.
        truth_data['Person'].append(person)
        truth_data['Date'].append(parse_date_final)
        truth_data['Veracity'].append(parse_meter_final)
        truth_data['Text'].append(parse_text_final)

    return truth_data

# Set up a dictionary to contain scraped data.
truth_data = {'Person':[], 'Date':[], 'Veracity':[], 'Text':[]}

# create a list of politician names from the lookup dictionary.
person_list = list(person_lookup_dict.keys())

# for each item in the list, return the HTML element containing the data I wish to scrape.
for person in person_list:
    show_progress(person,person_list)

    # Retrieve the list of URLs generated earlier of each person in the list
    url_list = person_lookup_dict[person]['urllist']

    # Iterate through the list of URLs.
    for url in url_list:
        # Retrieve the webpage for each URL.
        page = requests.get(url)

        # Wait a random amount of time between 10 and 20 seconds.
        # If an error is returned, state the status code and break the loop.
        sleep(randint(10,20))
        if page.status_code != 200:
            print('We may have a problem here.', page.status_code)
            break

        # Parse the HTML using BeatifulSoup.
        soup = BeautifulSoup(page.text, 'html.parser')

        # Save the elements containing the desired data to a list.
        fact_checks = soup.find_all(class_="statement__source", text=person)

        # Pass that list to the fuction "truth_extractor".
        truth_extractor(person, fact_checks, truth_data)   

        # Show progress.
        show_progress(url,url_list)
```


```python
# Convert dictionary of scraped data to dataframe.
truth_df = pd.DataFrame.from_dict(truth_data, orient="columns")

# Show final product.
truth_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Person</th>
      <th>Text</th>
      <th>Veracity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-04-10</td>
      <td>Donald Trump</td>
      <td>\nEPA administrator Scott Pruitt's short-term ...</td>
      <td>Mostly False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-04-10</td>
      <td>Donald Trump</td>
      <td>\nSays Scott Pruitt’s security spending was "s...</td>
      <td>Mostly False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-04-09</td>
      <td>Donald Trump</td>
      <td>\n"When a car is sent to the United States fro...</td>
      <td>Mostly True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-04-09</td>
      <td>Donald Trump</td>
      <td>\n"This will be the last time — April — that y...</td>
      <td>Mostly False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-04-06</td>
      <td>Donald Trump</td>
      <td>\n"In many places, like California, the same p...</td>
      <td>Pants on Fire!</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2018-04-04</td>
      <td>Donald Trump</td>
      <td>\n"We’ve started building the wall."\r\n\n</td>
      <td>Mostly False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2018-04-02</td>
      <td>Donald Trump</td>
      <td>\nSays caravans of people are coming to cross ...</td>
      <td>Half-True</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2018-04-02</td>
      <td>Donald Trump</td>
      <td>\nMexico has "very strong border laws -- ours ...</td>
      <td>Mostly False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2018-04-02</td>
      <td>Donald Trump</td>
      <td>\n"Only fools, or worse, are saying that our m...</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2018-03-28</td>
      <td>Donald Trump</td>
      <td>\n"Last year we lost $500 billion on trade wit...</td>
      <td>Mostly False</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2018-03-22</td>
      <td>Donald Trump</td>
      <td>\nSays Conor Lamb "ran on a campaign that said...</td>
      <td>False</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2018-03-21</td>
      <td>Donald Trump</td>
      <td>\nRobert Mueller’s investigative team has "13 ...</td>
      <td>Half-True</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2018-03-16</td>
      <td>Donald Trump</td>
      <td>\nSays Democratic obstruction is the reason wh...</td>
      <td>Half-True</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2018-03-15</td>
      <td>Donald Trump</td>
      <td>\nIn Japan, "they take a bowling ball from 20 ...</td>
      <td>False</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2018-03-15</td>
      <td>Donald Trump</td>
      <td>\n"We do have a Trade Deficit with Canada, as ...</td>
      <td>Mostly False</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2018-03-14</td>
      <td>Donald Trump</td>
      <td>\nSays China and Singapore impose the death pe...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2018-03-13</td>
      <td>Donald Trump</td>
      <td>\n"The state of California is begging us to bu...</td>
      <td>Pants on Fire!</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2018-03-13</td>
      <td>Donald Trump</td>
      <td>\nSays the U.S. steel and aluminum industry is...</td>
      <td>Mostly True</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2018-03-12</td>
      <td>Donald Trump</td>
      <td>\nThe last private rocket launch "cost $80 mil...</td>
      <td>Half-True</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2018-03-09</td>
      <td>Donald Trump</td>
      <td>\nAmerican aluminum and steel "are vital to ou...</td>
      <td>Half-True</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2018-03-09</td>
      <td>Donald Trump</td>
      <td>\n"When I was campaigning, I was talking about...</td>
      <td>False</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2018-03-07</td>
      <td>Donald Trump</td>
      <td>\n"Democrats are nowhere to be found on DACA."...</td>
      <td>False</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2018-03-06</td>
      <td>Donald Trump</td>
      <td>\nThe 2018 Academy Awards show was the "lowest...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2018-03-01</td>
      <td>Donald Trump</td>
      <td>\n"You take Pulse nightclub. If you had one pe...</td>
      <td>False</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2018-02-20</td>
      <td>Donald Trump</td>
      <td>\n"I have been much tougher on Russia than Oba...</td>
      <td>Mostly False</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2018-02-19</td>
      <td>Donald Trump</td>
      <td>\n"I never said Russia did not meddle in the e...</td>
      <td>Pants on Fire!</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2018-02-08</td>
      <td>Donald Trump</td>
      <td>\n"The Democrats are pushing for Universal Hea...</td>
      <td>Mostly False</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2018-02-07</td>
      <td>Donald Trump</td>
      <td>\nMany gang members have taken advantage of "g...</td>
      <td>Half-True</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2018-02-06</td>
      <td>Donald Trump</td>
      <td>\nAt the State of the Union address, Democrats...</td>
      <td>Pants on Fire!</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2018-02-02</td>
      <td>Donald Trump</td>
      <td>\n"Instead of two for one, we have cut 22 burd...</td>
      <td>Mostly False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1321</th>
      <td>2013-05-28</td>
      <td>Mitch McConnell</td>
      <td>\n\r\n\tSays Health and Human Services Secreta...</td>
      <td>Mostly True</td>
    </tr>
    <tr>
      <th>1322</th>
      <td>2010-06-14</td>
      <td>Mitch McConnell</td>
      <td>\n"A major part" of the climate change bill sp...</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1323</th>
      <td>2010-04-20</td>
      <td>Mitch McConnell</td>
      <td>\nNew financial regulation "actually guarantee...</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1324</th>
      <td>2010-02-19</td>
      <td>Mitch McConnell</td>
      <td>\nThe stimulus includes  "$219,000 to study  t...</td>
      <td>Half-True</td>
    </tr>
    <tr>
      <th>1325</th>
      <td>2010-02-19</td>
      <td>Mitch McConnell</td>
      <td>\n"$100,000 in  stimulus funds (were) used for...</td>
      <td>Mostly True</td>
    </tr>
    <tr>
      <th>1326</th>
      <td>2010-02-01</td>
      <td>Mitch McConnell</td>
      <td>\nOn a bipartisan task force on ways to improv...</td>
      <td>Full Flop</td>
    </tr>
    <tr>
      <th>1327</th>
      <td>2009-12-02</td>
      <td>Mitch McConnell</td>
      <td>\nThe Senate health care bill does not contain...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1328</th>
      <td>2009-06-01</td>
      <td>Mitch McConnell</td>
      <td>\n"The Department of Justice, under the Obama ...</td>
      <td>Half-True</td>
    </tr>
    <tr>
      <th>1329</th>
      <td>2009-05-19</td>
      <td>Mitch McConnell</td>
      <td>\nA public option for health care would end pr...</td>
      <td>Mostly False</td>
    </tr>
    <tr>
      <th>1330</th>
      <td>2009-03-03</td>
      <td>Mitch McConnell</td>
      <td>\n"In just one month, the Democrats have spent...</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1331</th>
      <td>2009-02-03</td>
      <td>Mitch McConnell</td>
      <td>\n To give the proposed economic stimulus plan...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1332</th>
      <td>2009-01-05</td>
      <td>Mitch McConnell</td>
      <td>\nIf Obama's economic plan creates 600,000 new...</td>
      <td>Mostly True</td>
    </tr>
    <tr>
      <th>1333</th>
      <td>2017-03-27</td>
      <td>Charles Schumer</td>
      <td>\n"In fact, if you add up the net wealth of hi...</td>
      <td>Mostly True</td>
    </tr>
    <tr>
      <th>1334</th>
      <td>2017-10-08</td>
      <td>Charles Schumer</td>
      <td>\nTrump’s tax plan is "completely focused on t...</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1335</th>
      <td>2017-10-06</td>
      <td>Charles Schumer</td>
      <td>\n"The Republicans are proposing to pay for th...</td>
      <td>Half-True</td>
    </tr>
    <tr>
      <th>1336</th>
      <td>2017-07-25</td>
      <td>Charles Schumer</td>
      <td>\n"When the price for oil goes up on the marke...</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1337</th>
      <td>2017-05-18</td>
      <td>Charles Schumer</td>
      <td>\n"President Obama became the first president ...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1338</th>
      <td>2017-03-27</td>
      <td>Charles Schumer</td>
      <td>\n"In fact, if you add up the net wealth of hi...</td>
      <td>Mostly True</td>
    </tr>
    <tr>
      <th>1339</th>
      <td>2017-01-27</td>
      <td>Charles Schumer</td>
      <td>\nSays Rex "Tillerson won't divest from Exxon....</td>
      <td>Pants on Fire!</td>
    </tr>
    <tr>
      <th>1340</th>
      <td>2017-01-10</td>
      <td>Charles Schumer</td>
      <td>\nSays Donald Trump campaigned on not cutting ...</td>
      <td>Mostly True</td>
    </tr>
    <tr>
      <th>1341</th>
      <td>2016-06-15</td>
      <td>Charles Schumer</td>
      <td>\nLast year, "244 suspected terrorists walked ...</td>
      <td>Mostly True</td>
    </tr>
    <tr>
      <th>1342</th>
      <td>2015-05-18</td>
      <td>Charles Schumer</td>
      <td>\n"It is simply a fact that insufficient fundi...</td>
      <td>Half-True</td>
    </tr>
    <tr>
      <th>1343</th>
      <td>2015-03-08</td>
      <td>Charles Schumer</td>
      <td>\n\r\n"The State Department asked all secretar...</td>
      <td>Mostly False</td>
    </tr>
    <tr>
      <th>1344</th>
      <td>2014-12-04</td>
      <td>Charles Schumer</td>
      <td>\nIn 2010, uninsured voters made up "about 5 p...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1345</th>
      <td>2014-05-08</td>
      <td>Charles Schumer</td>
      <td>\nIf you work 40 hours a week at the proposed ...</td>
      <td>Half-True</td>
    </tr>
    <tr>
      <th>1346</th>
      <td>2013-10-08</td>
      <td>Charles Schumer</td>
      <td>\nBecause of the 2011 debt ceiling fight, "the...</td>
      <td>Mostly True</td>
    </tr>
    <tr>
      <th>1347</th>
      <td>2010-08-04</td>
      <td>Charles Schumer</td>
      <td>\n\r\n\t"Eight of the nine justices in the Sup...</td>
      <td>Mostly True</td>
    </tr>
    <tr>
      <th>1348</th>
      <td>2010-04-13</td>
      <td>Charles Schumer</td>
      <td>\n"No one questioned that she (Judge Sotomayor...</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1349</th>
      <td>2010-01-22</td>
      <td>Charles Schumer</td>
      <td>\n"With a stroke of a pen, the (U.S. Supreme C...</td>
      <td>Mostly False</td>
    </tr>
    <tr>
      <th>1350</th>
      <td>2009-03-12</td>
      <td>Charles Schumer</td>
      <td>\n\r\n\t"No Bridge to Nowhere could occur."\n</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>1351 rows × 4 columns</p>
</div>




```python
# Save dataframe to CSV file.
truth_df.to_csv('politic_truth.csv', index=False)
```
