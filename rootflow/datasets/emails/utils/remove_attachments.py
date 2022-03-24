from email.mime.text import MIMEText
import mailbox
from tqdm import tqdm
import zarr
from correct_mbox_format import correct_mbox_format
import sys
    
#recursive function that removes both inline and normal attachments
def remove_attachments_from_mbox_string(mbox_string):
    mbox_message = mailbox.mboxMessage(mbox_string)
    
    #only messages that are multipart have attachments
    for i, part in enumerate(mbox_message.get_payload()):
        if not isinstance(part, str):
            if part.is_multipart():
                part, was_changed = remove_attachments_from_mbox_message(part)
                if was_changed:
                    payload = mbox_message.get_payload()
                    payload[i] = MIMEText('Attachment has been removed')
                    #mbox_message.set_payload(payload)

            if part.get_content_disposition() in ['inline', 'attachment']:
                #remove attachment
                payload = mbox_message.get_payload()
                payload[i] = MIMEText('Attachment has been removed')
                mbox_message.set_payload(payload)


    return mbox_message.as_string()

def remove_attachments_from_mbox_message(mbox_message):
    # #only messages that are multipart have attachments
    # if mbox_message.is_multipart():
    was_changed = False
    for i, part in enumerate(mbox_message.get_payload()):
        if not isinstance(part, str):
            if part.is_multipart():
                part, was_changed = remove_attachments_from_mbox_message(part)
                if was_changed:
                    payload = mbox_message.get_payload()
                    payload[i] = MIMEText('Attachment has been removed')
                    #mbox_message.set_payload(payload)

            if part.get_content_disposition() in ['inline', 'attachment']:
                #remove attachment
                payload = mbox_message.get_payload()
                payload[i] = MIMEText('Attachment has been removed')
                mbox_message.set_payload(payload)

    return mbox_message, was_changed

def check_mbox_message_part_for_attachment(mbox_part):
    if mbox_part.get_content_disposition() in ['inline', 'attachment']:
        return True
    
    return False

if __name__ == "__main__":
    FILE_NAME = "/mnt/3913be04-1a62-4a3d-b5c4-b804c51bfe73/branch/datasets/emails/zarr/emails.zarr"
    DATA_DELIMITER = "$$$data-separator$$$"
    zarr_file = zarr.open(FILE_NAME, mode='r+')

    sys.setrecursionlimit(50000)
    bytes_saved = 0
    for i, encoded_string in tqdm(enumerate(zarr_file), total=len(zarr_file), smoothing=.9):
        #remove attachments
        decoded_string = encoded_string.split(DATA_DELIMITER)

        if len(decoded_string) > 1:
            mbox_string = decoded_string[1]
            mbox_string = correct_mbox_format(mbox_string)
        else:
            mbox_string = decoded_string[0]

        mbox_no_attachments = remove_attachments_from_mbox_string(mbox_string)
        bytes_saved += len(mbox_string) - len(mbox_no_attachments)

        zarr_file[i] = mbox_no_attachments
        # print(f'File ID: {decoded_string[0]}')
        # print(f'Bytes_saved: {len(mbox_string) - len(mbox_no_attachments)}')

    print(f"We saved {bytes_saved} by cutting out attachments")


