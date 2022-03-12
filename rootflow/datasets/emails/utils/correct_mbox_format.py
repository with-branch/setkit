from ast import literal_eval

def correct_mbox_format(mbox_string):
    # mbox_string = mbox_string[2:]
    # mbox_string = mbox_string.replace("\\r\\n", "\n")
    # mbox_string = mbox_string.replace("id\n", "id")
    # mbox_string = mbox_string.replace("d=google.com;\n", "d=google.com;")
    # mbox_string = mbox_string.replace("\\'", "\'")
    # mbox_string = mbox_string.replace("\n <", " <")

    return literal_eval(mbox_string).decode("utf-8")