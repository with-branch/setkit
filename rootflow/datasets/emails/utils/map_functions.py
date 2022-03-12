from ast import literal_eval
from email.charset import BASE64
import copy
import email
import mailbox
import io
import os
from tqdm import tqdm
import json
import re
import quopri
import base64
import binascii
from uuid import uuid4
from bs4 import BeautifulSoup
from email.errors import HeaderParseError
from email import charset as _charset
Charset = _charset.Charset

from email.header import make_header

NL = '\n'
SPACE = ' '
BSPACE = b' '
SPACE8 = ' ' * 8
EMPTYSTRING = ''
MAXLINELEN = 78
FWS = ' \t'

ecre = re.compile(r'''
  =\?                   # literal =?
  (?P<charset>[^?]*?)   # non-greedy up to the next ? is the charset
  \?                    # literal ?
  (?P<encoding>[qQbB])  # either a "q" or a "b", case insensitive
  \?                    # literal ?
  (?P<encoded>.*?)      # non-greedy up to the next ?= is the encoded string
  \?=                   # literal ?=
  ''', re.VERBOSE | re.MULTILINE)

def decode_header(header):
    """Copy of the header decoding from email.header, with the exception that local encoded charsets do not fail.
    """
    # If it is a Header object, we can just return the encoded chunks.
    if hasattr(header, '_chunks'):
        return [(_charset._encode(string, str(charset)), str(charset))
                    for string, charset in header._chunks]
    # If no encoding, just return the header with no charset.
    if not ecre.search(header):
        return [(header, None)]
    # First step is to parse all the encoded parts into triplets of the form
    # (encoded_string, encoding, charset).  For unencoded strings, the last
    # two parts will be None.
    words = []
    for line in header.splitlines():
        parts = ecre.split(line)
        first = True
        while parts:
            unencoded = parts.pop(0)
            if first:
                unencoded = unencoded.lstrip()
                first = False
            if unencoded:
                words.append((unencoded, None, None))
            if parts:
                charset = parts.pop(0).lower()
                encoding = parts.pop(0).lower()
                encoded = parts.pop(0)
                charset = charset.split("*")[0] # Hoping that local is the only time that a star is included in the charset (e.g. utf-8*ja)
                words.append((encoded, encoding, charset))
    # Now loop over words and remove words that consist of whitespace
    # between two encoded strings.
    droplist = []
    for n, w in enumerate(words):
        if n>1 and w[1] and words[n-2][1] and words[n-1][0].isspace():
            droplist.append(n-1)
    for d in reversed(droplist):
        del words[d]

    # The next step is to decode each encoded word by applying the reverse
    # base64 or quopri transformation.  decoded_words is now a list of the
    # form (decoded_word, charset).
    decoded_words = []
    for encoded_string, encoding, charset in words:
        if encoding is None:
            # This is an unencoded word.
            decoded_words.append((encoded_string, charset))
        elif encoding == 'q':
            word = email.quoprimime.header_decode(encoded_string)
            decoded_words.append((word, charset))
        elif encoding == 'b':
            paderr = len(encoded_string) % 4   # Postel's law: add missing padding
            if paderr:
                encoded_string += '==='[:4 - paderr]
            try:
                word = email.base64mime.decode(encoded_string)
            except binascii.Error:
                raise HeaderParseError('Base64 decoding error')
            else:
                decoded_words.append((word, charset))
        else:
            raise AssertionError('Unexpected encoding: ' + encoding)
    # Now convert all words to bytes and collapse consecutive runs of
    # similarly encoded words.
    collapsed = []
    last_word = last_charset = None
    for word, charset in decoded_words:
        if isinstance(word, str):
            word = bytes(word, 'raw-unicode-escape')
        if last_word is None:
            last_word = word
            last_charset = charset
        elif charset != last_charset:
            collapsed.append((last_word, last_charset))
            last_word = word
            last_charset = charset
        elif last_charset is None:
            last_word += BSPACE + word
        else:
            last_word += word
    collapsed.append((last_word, last_charset))
    return collapsed

def decode_quoted(input_string: str) -> str:
    bytes = quopri.decodestring(input_string.encode('utf-8'))
    try:
        output_string = bytes.decode('utf-8')
    except UnicodeDecodeError:
        output_string = bytes.decode('latin1')
    return output_string

def decode_base64(input_string: str) -> str:
    try:
        bytes = base64.b64decode(input_string)
    except ValueError:
        return input_string
    try:
        output_string = bytes.decode('utf-8')
    except UnicodeDecodeError:
        output_string = bytes.decode('latin1')
    return output_string

def decode_hex(input_string: str) -> str:
    return decode_base64(input_string)

def get_domain_from_url(url_string: str) -> str:
    domains = re.match(
        r'(?:https:\/\/|http:\/\/|www\.)+(\S*?)(?:\/|\s|\Z)',
        url_string
    )
    if domains is None:
        return ''
    domain_components = domains.group(1).split('.')
    if len(domain_components) == 0:
        return ''
    elif len(domain_components) == 1:
        return domain_components[-1]
    else:
        return domain_components[-2]

def format_url(input_url_string: str) -> str:
    domain = get_domain_from_url(input_url_string)
    return f'URL:{domain}'

def replace_urls(input_string: str) -> str:
    return re.sub(
        r'(https:\/\/|http:\/\/|www)\S*',
        lambda match: format_url(match.group(0)),
        input_string
    )

def remove_extra_whitespace(input_string: str) -> str:
    input_string = re.sub(r'[ \u200c\u00a0]{2,}', ' ', input_string)
    input_string = re.sub(r'( \n){2,}', '\n', input_string)
    input_string = re.sub(r'[ \r\n]{2,}', '\n', input_string)
    input_string = re.sub(r'(\n){2,}', '\n', input_string)
    return input_string

def clean_text(input_string: str) -> str:
    return remove_extra_whitespace(replace_urls(input_string))

def html_to_text(html_string: str) -> str:
    soup = BeautifulSoup(html_string, features = 'lxml')
    return soup.get_text()

UTF8_ENCODING_VARIANTS = ['8bit', '7bit', 'amazonses.com', 'utf-8', 'utf8', 'hexa']
BASE64_ENCODING_VARIANTS = ['base64', '64bit']
QUOTED_ENCODING_VARIANTS = ['quoted-printable']

def decode_plain(message: mailbox.mboxMessage):
    encoding = message.get('Content-Transfer-Encoding')
    if encoding is None:
        return message.get_payload()
    encoding = encoding.lower().replace(' ', '')
    if any([variant in encoding for variant in QUOTED_ENCODING_VARIANTS]):
        return decode_quoted(message.get_payload())
    elif any([variant in encoding for variant in BASE64_ENCODING_VARIANTS]):
        return html_to_text(decode_base64(message.get_payload()))
    elif any([variant in encoding for variant in UTF8_ENCODING_VARIANTS]):
        return message.get_payload()
    else:
        print(message.as_string())
        raise AttributeError(f'Content type: text/plain and Encoding: {encoding} is not handled')

def decode_html(message: mailbox.mboxMessage):
    encoding = message.get('Content-Transfer-Encoding')
    if encoding is None:
        return message.get_payload()
    encoding = encoding.lower().replace(' ', '')
    if any([variant in encoding for variant in BASE64_ENCODING_VARIANTS]):
        return decode_base64(message.get_payload())
    elif 'quoted-printable' in encoding:
        return decode_quoted(message.get_payload())
    elif any([variant in encoding for variant in UTF8_ENCODING_VARIANTS]):
        return message.get_payload()
    else:
        print(message.as_string())
        raise AttributeError(f'Content type: text/html and Encoding: {encoding} is not handled')

CONTENT_TYPE_PREFERENCE = ['multipart/mixed', 'text/html', 'multipart/alternative', 'text/plain']

def decode_mbox_email_message(message: mailbox.mboxMessage):
    content_type = message.get_content_type().lower()
    if 'multipart/alternative' in content_type or 'multipart/related' in content_type:
        messages = {}
        for msg in message.get_payload():
            try:
                messages[msg.get_content_type()] = msg
            except AttributeError as e:
                return (None, None, False) # for now, we can remove this later
        for content_type in CONTENT_TYPE_PREFERENCE:
            if content_type in messages.keys():
                return decode_mbox_email_message(messages[content_type])
        return (None, None, False)
    elif 'multipart/mixed' in content_type:
        message_text = ''
        display_content = ''
        display_is_html = False
        for msg in message.get_payload():
            text, display, is_html = decode_mbox_email_message(msg)
            if not text is None:
                message_text += text + '\n'
            if not display is None:
                display_content += display + '\n'
            if is_html:
                display_is_html = True
    elif 'text/plain' in content_type:
        message_text = decode_plain(message)
        display_content = message_text
        display_is_html = False
    elif 'text/html' in content_type:
        display_content = decode_html(message)
        message_text = html_to_text(display_content)
        display_is_html = True
    elif 'application/ics':
        return (None, None, False)
    else:
        print(message.as_string())
        raise AttributeError(f'Content type: {content_type} is not handled')
    return clean_text(message_text), display_content, display_is_html

def flatten_message(msg, from_addr=None, to_addrs=None,
                    mail_options=()):
    """Converts message to bytestring
    """
    # 'Resent-Date' is a mandatory field if the Message is resent (RFC 2822
    # Section 3.6.6). In such a case, we use the 'Resent-*' fields.  However,
    # if there is more than one 'Resent-' block there's no way to
    # unambiguously determine which one is the most recent in all cases,
    # so rather than guess we raise a ValueError in that case.

    resent = msg.get_all('Resent-Date')
    if resent is None:
        header_prefix = ''
    elif len(resent) == 1:
        header_prefix = 'Resent-'
    else:
        raise ValueError("message has more than one 'Resent-' header block")
    if from_addr is None:
        # Prefer the sender field per RFC 2822:3.6.2.
        from_addr = (msg[header_prefix + 'Sender']
                        if (header_prefix + 'Sender') in msg
                        else msg[header_prefix + 'From'])
        from_addr = email.utils.getaddresses([from_addr])[0][1]
    if to_addrs is None:
        addr_fields = [f for f in (msg[header_prefix + 'To'],
                                    msg[header_prefix + 'Bcc'],
                                    msg[header_prefix + 'Cc'])
                        if f is not None]
        to_addrs = [a[1] for a in email.utils.getaddresses(addr_fields)]
    # Make a local copy so we can delete the bcc headers.
    msg_copy = copy.copy(msg)
    del msg_copy['Bcc']
    del msg_copy['Resent-Bcc']
    international = False
    try:
        ''.join([from_addr, *to_addrs]).encode('ascii')
    except UnicodeEncodeError:
        international = True
    with io.BytesIO() as bytesmsg:
        if international:
            g = email.generator.BytesGenerator(
                bytesmsg, policy=msg.policy.clone(utf8=True))
            mail_options = (*mail_options, 'SMTPUTF8', 'BODY=8BITMIME')
        else:
            g = email.generator.BytesGenerator(bytesmsg)
        g.flatten(msg_copy, linesep='\r\n')
        flatmsg = bytesmsg.getvalue()
    return str(flatmsg)

def format_mbox_email_item(email: mailbox.mboxMessage) -> dict:
    data, display, is_html = decode_mbox_email_message(email)
    email_sender = email['FROM']
    if not email_sender is None:
        email_sender = str(make_header(decode_header(email_sender)))
    subject = email['SUBJECT']
    if not subject is None:
        try:
            subject = str(make_header(decode_header(subject)))
        except LookupError as e:
            print(subject)
            
    return {
        'from' : email_sender,
        'subject' : subject,
        'text' : data,
        'display' : display,
        'mbox' : flatten_message(email),
        "display_is_html" : is_html
    }

def format_mbox(mbox_file):
    email_file_paths = []
    mbox_obj = mailbox.mbox(os.path.abspath(mbox_file))
    total_emails = len(mbox_obj)
    disc_rep_counter = 0

    discrep_list = []
    for email in tqdm(mbox_obj, total=total_emails):
        try:
            formatted_email = format_mbox_email_item(email)
            mbox_string = formatted_email["mbox"]
            # mbox_string = mbox_string[2:]
            # mbox_string = mbox_string.replace("\\r\\n", "\n")
            # mbox_string = mbox_string.replace("id\n", "id")
            # mbox_string = mbox_string.replace("d=google.com;\n", "d=google.com;")
            # mbox_string = mbox_string.replace("\\'", "\'")
            # mbox_string = mbox_string.replace("\n <", " <")

            mbox_string = literal_eval(mbox_string).decode("utf-8")

            mbox_message = mailbox.mboxMessage(mbox_string)

            # temp_from = mbox_message['FROM']
            # temp_from = temp_from.replace("\n", "")
            # temp_from = temp_from.replace("\t", "")
            # temp_from = temp_from.replace("\r", "")
            # temp_from = temp_from.replace(" ", "")

            # correct_from = email["FROM"]
            # correct_from = correct_from.replace("\n", "")
            # correct_from = correct_from.replace("\t", "")
            # correct_from = correct_from.replace("\r", "")
            # correct_from = correct_from.replace(" ", "")

            # if temp_from != correct_from:
            #     discrep_list.append((correct_from, temp_from))

            if mbox_message["FROM"] == None:
                disc_rep_counter += 1
            # else:
            #     f = open("tanner", "w")
            #     f.write(mbox_string)
            #     f.close()

            #     f = open("correct", "w")
            #     f.write(email.as_string())
            #     f.close()

            #     break
        except Exception as e:
            print(e)
            pass
    print(f'Formatted mbox file {mbox_file}')
    print(f'{disc_rep_counter} emails out of {len(mbox_obj)} were incorrect')
    return email_file_paths

if __name__ == "__main__":
    MBOX_FILE_PATH = "/mnt/3913be04-1a62-4a3d-b5c4-b804c51bfe73/branch/datasets/emails/Takeout/Mail/All mail Including Spam and Trash.mbox"

    format_mbox(MBOX_FILE_PATH)