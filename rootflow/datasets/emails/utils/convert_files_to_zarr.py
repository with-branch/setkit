from genericpath import isfile
import os
import time
from tqdm import tqdm
import json
import zarr

LOCAL_FILE_PATH="/home/dallin/Branch/Datasets/Emails"
DESTINATION_FILE_PATH="/media/dallin/Linux_2/branch/datasets/emails/zarr/emails.zarr"
ZARR_DELIMITER="$$$data-separator$$$"
CHUNK_SIZE=50

num_files = len(os.listdir(LOCAL_FILE_PATH))
if os.path.exists(DESTINATION_FILE_PATH):
    #over-write existing zarr
    zarr_file = zarr.open(DESTINATION_FILE_PATH, mode='a', shape=num_files, chunks=CHUNK_SIZE, dtype=str)    
else:
    #create a new zarr
    store = zarr.NestedDirectoryStore(DESTINATION_FILE_PATH)
    zarr_file = zarr.create(shape=num_files, chunks=CHUNK_SIZE, 
        store=store, dtype=str)

#store formatted strings here so we can later append to the zar as a group
#this decreases resize time
temp_store = []

#setup index counters for store and zarr
store_index = 0
zarr_initial_index = 0
file_load_error_counter=0

#loop through all json files, format the data inside, and append the formatted
#data to the zarr
for i, file in tqdm(enumerate(os.listdir(LOCAL_FILE_PATH)), total=num_files):

    path_to_file = os.path.join(LOCAL_FILE_PATH, file)
    if isfile(path_to_file):

        #format data from file
        with open(path_to_file, 'r') as file_object:
                         
            #some files may be corrupted
            try:
                #load the json data into a python dict
                item_data = json.load(file_object)

                #cannot concatonate None to a string
                subject = item_data["data"]["subject"]
                if subject is None:
                    subject = ""
                label = item_data["label_info"]["label"]
                if label is None:
                    label = ""
                elif label:
                    label = "True"
                else:
                    label = "False"

                #convert everything to a delimited string
                # id ZARR_DELIMITER mbox ZARR_DELIMITER label ZARR_DELIMITER oracle_id ZARR_DELIMITER dataset_id
                zarr_string = item_data["label_info"]["example_id"]
                zarr_string += ZARR_DELIMITER + item_data["data"]["mbox"]
                zarr_string += ZARR_DELIMITER + label + ZARR_DELIMITER + item_data["label_info"]["oracle_id"]
                zarr_string += ZARR_DELIMITER + item_data["label_info"]["dataset_id"]             

                if len(temp_store) < CHUNK_SIZE:
                    temp_store.append(str(zarr_string))
                    store_index += 1
                elif len(temp_store) == CHUNK_SIZE and store_index < CHUNK_SIZE:
                    temp_store[store_index] = zarr_string
                    store_index += 1
                else:
                    #insert full chunk into zarr
                    zarr_file[zarr_initial_index:i-file_load_error_counter] = temp_store
                    zarr_initial_index = i-file_load_error_counter
                    temp_store[0] = zarr_string
                    store_index = 1               

            except json.JSONDecodeError:
                file_load_error_counter += 1

            if i == num_files - 1:
                    #could have a partial chunk so we insert now
                    zarr_file[zarr_initial_index:i+1-file_load_error_counter] = temp_store[0:store_index]
                    zarr_file.resize(len(zarr_file) - file_load_error_counter)

print(f"There were {file_load_error_counter} files that could not be loaded")



