from __future__ import division
import pickle

data_dict = pickle.load(open("data/final_project_dataset.pkl", "r") )
# Remove outliers:
data_dict.pop('TOTAL', None)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', None)
# Add that one feature:
for key,value in data_dict.iteritems():
    if value['from_this_person_to_poi'] == 'NaN':
        value['from_this_person_to_poi'] = 0
    if value['from_poi_to_this_person'] == 'NaN':
        value['from_poi_to_this_person'] = 0
    if value['to_messages'] == 'NaN':
        value['to_messages'] = 0
    if value['from_messages'] == 'NaN':
        value['from_messages'] = 0
    if (value['to_messages'] + value['from_messages'] == 0):
        value['email_ratio_with_poi'] = 0
    else:
        value['email_ratio_with_poi'] = (int(value['from_this_person_to_poi'])+\
                                     int(value['from_poi_to_this_person']))/\
                                     (int(value['to_messages'])+\
                                     int(value['from_messages']))

pickle.dump(data_dict, open('data/own_data_dict.pkl', "w") )