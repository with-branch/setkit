
            data_in_memory = []
            error_counter = 0
            complete_attachment_time = 0
            inner_time = 0
            for file in os.listdir(self.LOCAL_FILE_PATH):
                path_to_file = os.path.join(self.LOCAL_FILE_PATH, file)
                if isfile(path_to_file):
                    #format data from file
                    with open(path_to_file, 'r') as file_object:             
                        #some files may be corrupted
                        try:
                            item_data = json.load(file_object)

                            #remove attachments
                            mbox_string = item_data["data"]["mbox"]
                            start = time.time()
                            mbox_data, inn_tot = remove_attachments_from_mbox_string(mbox_string)
                            end = time.time()
                            inner_time += inn_tot
                            tot = end-start
                            complete_attachment_time += tot

                            #format data
                            formatted_data = {"id": item_data["label_info"]["example_id"], "data": mbox_data, "label": None}

                            #add data to list
                            data_in_memory.append(formatted_data)

                        except json.JSONDecodeError:
                            error_counter += 1

                        if len(data_in_memory) >= 100:
                            break

            print(f"There were {error_counter} files that could not be loaded")
            print(f"We successfully loaded {len(data_in_memory)} files")
            print(f"averge attachment time {complete_attachment_time / len(data_in_memory)}")
            print(f"averge inner attachment time {inner_time / len(data_in_memory)}")