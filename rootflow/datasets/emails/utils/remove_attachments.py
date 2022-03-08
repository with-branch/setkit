import mailbox
from tqdm import tqdm
import zarr
    
#recursive function that removes both inline and normal attachments
def remove_attachments_from_mbox_string(mbox_string):
    mbox_message = mailbox.mboxMessage(mbox_string)
    
    #only messages that are multipart have attachments
    for i, part in enumerate(mbox_message.walk()):
        if part.is_multipart():
            part = remove_attachments_from_mbox_message(part)
            mbox_message.get_payload()[i] = part.get_payload()

        if part.get_content_disposition() in ['inline', 'attachment']:
            #remove attachment
            mbox_message.get_payload()[i] = ""

    return mbox_message.as_string()

def remove_attachments_from_mbox_message(mbox_message):
    # #only messages that are multipart have attachments
    # if mbox_message.is_multipart():
    for i, part in enumerate(mbox_message.walk()):
        if part.is_multipart():
            part = remove_attachments_from_mbox_message(part)

        if part.get_content_disposition() in ['inline', 'attachment']:
            #remove attachment
            mbox_message.get_payload()[i] = ""

    return mbox_message

def check_mbox_message_part_for_attachment(mbox_part):
    if mbox_part.get_content_disposition() in ['inline', 'attachment']:
        return True
    
    return False

if __name__ == "__main__":
    FILE_NAME = "/media/dallin/Linux_2/branch/datasets/emails/zarr/emails.zarr"
    DATA_DELIMITER = "$$$data-separator$$$"
    zarr_file = zarr.open(FILE_NAME, mode='r')


    for i, encoded_string in tqdm(enumerate(zarr_file), total=len(zarr_file), smoothing=.9):
        if len(encoded_string) > 1_000_000:
            #remove attachments
            decoded_string = encoded_string.split(DATA_DELIMITER)

            mbox_string = decoded_string[1]
            f = open("before.json", "w")
            f.write(mbox_string)
            f.close()

            mbox_no_attachments = remove_attachments_from_mbox_string(mbox_string)
            f = open("after.json", "w")
            f.write(mbox_string)
            f.close()

            print(f'File ID: {decoded_string[0]}')
            print(f'Bytes_saved: {len(mbox_string) - len(mbox_no_attachments)}')


