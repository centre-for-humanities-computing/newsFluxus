import argparse
import nltk
import stanza

def main():

  ap = argparse.ArgumentParser(description="[INFO] language resources to install")
  ap.add_argument("-l", "--language", required=True, help="language code from Stanza")
  args = vars(ap.parse_args())

  nltk.download("stopwords")
  stanza.download(args["language"])

if __name__=="__main__":
    main()