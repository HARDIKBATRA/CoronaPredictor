from flask import Flask ,render_template,request
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests 
from bs4 import BeautifulSoup 
import geopandas as gpd
from prettytable import PrettyTable
app = Flask(__name__)
file=open('model.pkl','rb')
clf=pickle.load(file)
file.close()
@app.route('/',methods=["GET","POST"])
def hello_world():
    if request.method=="POST":
        myDict=request.form
        Fever=float(myDict['Fever'])
        BodyPain=int(myDict['BodyPain'])
        Age=int(myDict['Age'])
        RunnyNose=int(myDict['RunnyNose'])
        DifficultBreathing=int(myDict['DifficultBreathing'])
        inputFeatures=[Fever,BodyPain,Age,RunnyNose,DifficultBreathing]
        infProb=clf.predict_proba([inputFeatures])[0][1]
        print(infProb)


        return render_template('show.html',inf=round(infProb*100))
    
    return render_template('index.html')


url = 'https://www.mohfw.gov.in/'

web_content = requests.get(url).content

soup = BeautifulSoup(web_content, "html.parser")

extract_contents = lambda row: [x.text.replace('\n', '') for x in row]

stats = [] 
all_rows = soup.find_all('tr')
for row in all_rows:
    stat = extract_contents(row.find_all('td')) 

    if len(stat) == 5:
        stats.append(stat)

new_cols = ["Sr.No", "States/UT","Confirmed","Recovered","Deceased"]
state_data = pd.DataFrame(data = stats, columns = new_cols)
state_data['Confirmed'] = state_data['Confirmed'].map(int)
state_data['Recovered'] = state_data['Recovered'].map(int)
state_data['Deceased'] = state_data['Deceased'].map(int)

sns.set_style('ticks')
plt.figure(figsize = (15,10))
plt.barh(state_data['States/UT'],    state_data['Confirmed'].map(int),align = 'center', color = 'lightblue', edgecolor = 'blue')
plt.xlabel('No. of Confirmed cases', fontsize = 18)
plt.ylabel('States/UT', fontsize = 18)
plt.gca().invert_yaxis()
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.title('Total Confirmed Cases Statewise', fontsize = 18 )
for index, value in enumerate(state_data['Confirmed']):
    plt.text(value, index, str(value), fontsize = 12)
india_bar_graph=plt.savefig('static/india_bar.png')

group_size = [sum(state_data['Confirmed']),
              sum(state_data['Recovered']),
              sum(state_data['Deceased'])]
group_labels = ['Confirmed\n' + str(sum(state_data['Confirmed'])),
                'Recovered\n' + str(sum(state_data['Recovered'])),
                'Deceased\n' + str(sum(state_data['Deceased']))]
custom_colors = ['skyblue','yellowgreen','tomato']
plt.figure(figsize = (5,5))
plt.pie(group_size, labels = group_labels, colors = custom_colors)
central_circle = plt.Circle((0,0), 0.5, color = 'white')
fig = plt.gcf()
fig.gca().add_artist(central_circle)
plt.rc('font', size = 12)
india_pie_chart=plt.savefig('static/india_pie2.png')
map_data = gpd.read_file('Indian_States.shp')
map_data.rename(columns = {'st_nm':'States/UT'}, inplace = True)
map_data.head()

map_data['States/UT'] = map_data['States/UT'].str.replace('&', 'and')
map_data['States/UT'].replace('Arunanchal Pradesh', 'Arunachal Pradesh', inplace = True)
map_data['States/UT'].replace('Telangana', 'Telengana', inplace = True)
map_data['States/UT'].replace('NCT of Delhi', 'Delhi', inplace = True)

merged_data = pd.merge(map_data, state_data, how = 'left', on = 'States/UT')
merged_data.fillna(0, inplace = True)
merged_data.drop('Sr.No', axis = 1, inplace = True)
merged_data.head()

fig, ax = plt.subplots(1, figsize=(20, 12))
ax.axis('off')
ax.set_title('Covid-19 Statewise Data - Confirmed Cases', fontdict = {'fontsize': '25', 'fontweight' : '3'})

merged_data.plot(column = 'Confirmed', cmap='YlOrRd', linewidth=0.8, ax=ax, edgecolor='0.8', legend = True)
india=plt.savefig('static/india.png')
map_data = gpd.read_file('Indian_States.shp')

if __name__ == "__main__":
    app.run(debug=True)
