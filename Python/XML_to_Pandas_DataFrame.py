#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 01:57:35 2017

@author: tensorflow
"""
import xml.etree.ElementTree as ET
from lxml import etree
import pandas as pd
import os

def xml2df(xml_data):
    tree = ET.parse(xml_data) #Initiates the tree Ex: <user-agents>
    root = tree.getroot() #Starts the root of the tree Ex: <user-agent>
    all_records = [] #This is our record list which we will convert into a dataframe
    headers = [] #Subchildren tags will be parsed and appended here
    for i, child in enumerate(root): #Begin looping through our root tree
        record = [] #Place holder for our record
        for subchild in child: #iterate through the subchildren to user-agent, Ex: ID, String, Description.
            record.append(subchild.text) #Extract the text and append it to our record list
            if subchild.tag not in headers: #Check the header list to see if the subchild tag <ID>, <String>... is in our headers field. If not append it. This will be used for our headers.
                headers.append(subchild.tag)
        all_records.append(record) #Append this record to all_records.
    return pd.DataFrame(all_records, columns=headers) #Finally, return our Pandas dataframe with headers in the column.
    
dataframe_output = xml2df(xml_data)
