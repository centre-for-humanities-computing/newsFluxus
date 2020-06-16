import argparse
import nltk
import stanfordnlp

def main():

  ap = argparse.ArgumentParser(description="[INFO] language resources to install")
  ap.add_argument("-l", "--language", required=True, help="language code from StanfordNLP")
  args = vars(ap.parse_args())

  nltk.download("stopwords")
  stanfordnlp.download(args["language"])

if __name__=="__main__":
    main()