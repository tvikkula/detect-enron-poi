#!/usr/bin/python
import sys
sys.path.append( "./tools/" )
import poi_email_addresses
import parse_out_email_text
import pprint
def parseEmails():
    emailaddresses = poi_email_addresses.poiEmails()
    emaildata = {}
    for email in emailaddresses:
        with open('emails_by_address/' + email, 'r') as f:
            text = parse_out_email_text.parseOutText(f)
            emaildata[email] = text
    return emaildata

if __name__ == '__main__':
    pprint.pprint(parseEmails())