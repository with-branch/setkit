from google.cloud import storage
        file_path = os.path.join(directory, self.FILE_NAME)
        print(file_path)
        storage_client = storage.Client()

        #count how many files we are going to download
        bucket = storage.Bucket(storage_client, name=self.BUCKET) 
        file_names_iter = storage_client.list_blobs(bucket, prefix = self.prefix)
        num_files = sum(1 for blob in file_names_iter)

        if  not exists( file_path ):
            store = zarr.NestedDirectoryStore(file_path)
            self.data = zarr.create(shape=num_files, chunks=self.CHUNK_SIZE, 
                store=store, dtype=str)
        else:
            self.data = zarr.open(file_path, mode='a', shape=num_files, chunks=self.CHUNK_SIZE, dtype=str)

        
        print("Downloading files")
        file_names_iter = storage_client.list_blobs(bucket, prefix = self.prefix)
        #to keep the zarr of resizing so much we insert a chunk at a time
        #dynamic array could be switched with linked list to avoid resizing
        temp_store = []
        store_index = 0
        zarr_initial_index = 0
        bytes_loaded = 0
        for i, file in tqdm(enumerate(file_names_iter), total=num_files, smoothing=.9):
            data = file.download_as_string()
            json_object = json.loads(data)

            #formatt the data
            data_dict = {"from": json_object["data"]["from"], 
                "subject": json_object["data"]["subject"], "mbox": json_object["data"]["mbox"]}
            full_item = {"id": json_object["label_info"]["example_id"], "data": data_dict, 
            "target": json_object["label_info"]["label"], "oracle_id": json_object["label_info"]["oracle_id"], "group_id": json_object["label_info"]["dataset_id"]}
            bytes_loaded += len(full_item)
            # full_item = json_object["label_info"]["example_id"] + self.DATA_DELIMITER + json_object["data"]["from"] + self.DATA_DELIMITER
            # full_item += json_object["data"]["subject"] + self.DATA_DELIMITER + json_object["data"]["mbox"]

            if len(temp_store) < self.CHUNK_SIZE:
                temp_store.append(str(full_item))
                store_index += 1
            elif len(temp_store) == self.CHUNK_SIZE and store_index < self.CHUNK_SIZE:
                temp_store[store_index] = full_item
                store_index += 1
            else:
                #insert full chunk into zarr
                self.data[zarr_initial_index:i] = temp_store
                zarr_initial_index = i
                temp_store[0] = full_item
                store_index = 1

            if i == num_files - 1:
                #could have a partial chunk so we insert now
                self.data[zarr_initial_index:i+1] = temp_store[0:store_index]