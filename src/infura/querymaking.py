'''
Extract subset from one raw news-file.

If your data is in a single folder:
- and you want to save subset in a single file
- and you want to keep the original file structure

If your data is in multiple folders:
- and you want to save subset in a single file
- and you want to save subset in one file per folder
- and you want to keep the original file structure

IDEA
another regex pattern for filtering file paths (that way some newspapers can be removed)
'''
# %%

import os
import re
import glob
import datetime

import ndjson
import numpy as np
import pandas as pd
from tqdm import tqdm
from mdutils.mdutils import MdUtils
from ciso8601 import parse_datetime_as_naive, parse_datetime
# probalby need to swtich to parse_datetime once UK/US newspapers start coming in

from tekisuto.preprocessing.regxfilter import RegxFilter
from .util import resolve_path, flag_fuzzy_duplicates

# %%
class Query:
    '''
    Set everything up.
    '''
    def __init__(self,
                query_pattern,
                date_start,
                date_end,
                paths=None,
                paths_pattern=False,
                condition_pattern=None,
                export_dir=None,
                export_file_ext='ndjson',
                fields_to_match=None,
                limited_export=False) -> None:
        '''Generic query settings

        Parameters
        ----------
        query_pattern : re.Pattern
            Regex pattern to look for in the texts
        date_start : str
            When to start matching (this date included).
            Format: so that `parse_datetime` can make sense of it.
            e.g. YYYY-MM-DD
        date_end : str
            When to stop matching (this date included).
            Same format considerations as `date_start`.
        paths : list|str, optional
            List of paths, or a file-finding glob pattern, by default None
        paths_pattern : bool, optional
            Is `paths` a filefinding pattern By default False
        condition_pattern : re.Pattern, optional
            Regex pattern to look for in adition to `query_pattern`, by default None
        export_dir : str, optional
            Path to directory where subset is to be saved, by default None
        export_file_ext : str, optional
            File extension / format to export the results in, by default 'ndjson'
        fields_to_match : list|tuple, optional
            Names of fields where to run regex, by default None
        limited_export : bool|list|tuple, optional
            Pick only some fields to export? By default False.
            Specify field names to export if True.
        '''
        
        # FIELDS
        self.fields_to_match = self.validate_match_fields(fields_to_match)
        if limited_export:
            self.limited_export = self.validate_match_fields(limited_export)
        else:
            self.limited_export = False

        # DATES
        self.date_start = self.parse_str_date(date_start)
        self.date_end = self.parse_str_date(date_end)
            # bool: hourly resolution of query?
        self.finegrained_dates = self.is_finegrained_time()

        # REGEX
            # main pattern
        self.query_pattern = query_pattern
            # conditional pattern
        if condition_pattern:
            self.condition_pattern = condition_pattern
        else:
            self.condition_pattern = None

        # input dataset (paths)
        if paths_pattern:
            paths = glob.glob(paths)

        else:
            paths = self.validate_paths(paths)
        
        # making sure they are sorted
        paths = self.validate_paths(paths)
        self.paths = sorted(paths)

        # output paths
        if export_dir:
            self.export_dir = resolve_path(export_dir)
        
        # export format
        self.file_ext = export_file_ext


    # INPUT VALIDATION
    @staticmethod
    def validate_paths(path_list) -> list:
        '''
        Make sure paths exist and are a list (iterable).
        '''
        # FIXME unsmooth checking for empty
        if not path_list:
            raise ValueError('No files to process found! Check `paths` in input.')

        if isinstance(path_list, str):
            path_list = [path_list]
        
        path_list = [resolve_path(path) for path in path_list]

        return path_list


    @staticmethod
    def parse_str_date(str_date) -> datetime.datetime:
        '''
        Parse str to datetime.datetime
        '''
        return parse_datetime_as_naive(str_date)

    
    def is_finegrained_time(self) -> bool:
        '''
        If either date_start or date_end have information on hours/minutes,
        timeframe to evaluate will be considered finegrained.

        This slows the pipeline a bit, as timestamps must be checked inside article,
        in addition to timestamps witch daily resolution in filenames.
        '''
        start_grain = self.date_start.hour + self.date_start.minute
        end_grain = self.date_end.hour + self.date_end.minute

        if start_grain == 0 and end_grain == 0:
            return False
        else:
            return True

    @staticmethod
    def validate_match_fields(fields_to_match) -> tuple:
        '''
        Turns inputed match_fields into a tuple for faster runing.

        TODO 
        - correct another datatypes?
        - check if they are in the dataset (that would mean moving this method to another class)
            - might have to load a sample of the dataset to check
        '''
        if isinstance(fields_to_match, list):
            fields_to_match = tuple(fields_to_match)
        
        return fields_to_match

    # GENERIC PATH MANIPULATION
    @staticmethod
    def validate_string(string) -> str:
        if not isinstance(string, str):
            string = str(string)
        
        return string

    @staticmethod
    def basepath_to_name(path, ext=None) -> str:
        '''Get the name of the deepest object in a file path.

        Parameters
        ----------
        path : str
            A path to file or folder
        ext : str, optional
            If specified, add a file extension to extracted name, by default None
        '''
        if ext:
            return os.path.basename(os.path.normpath(path)) + '.' + ext
        else:
            return os.path.basename(os.path.normpath(path))

    # GENERIC EXTRACTION METHODS
    @staticmethod
    def dated(date, date_start, date_end) -> bool:
        '''date within range?
        '''
        return date >= date_start and date <= date_end

    @staticmethod
    def matched(text, pattern) -> bool:
        '''pattern detected?
        '''
        return bool(pattern.search(text))

    def flag_matched_articles(self, content_list, pattern) -> list:
        '''returns a list of bools
        '''
        return [self.matched(article, pattern) for article in content_list]

    @staticmethod
    def extract_matched(base_list, flag_list) -> list:
        '''returns a list of content (e.g. dict) 
        '''
        assert len(base_list) == len(flag_list)
        return [x for i, x in enumerate(base_list) if flag_list[i]]

    @staticmethod
    def find_mutual_trues(list1, list2) -> np.ndarray:
        '''
        Flag indices where BOTH lists have Ture

        Parameters
        ----------
        list1, list2 : bool
            lists of bool of the same length
        '''
        assert len(list1) == len(list2)

        if list1 and list2:
            list1 = np.array(list1)
            list2 = np.array(list2)

            return list1 & list2

        else:
            return None


    # I/O
    @staticmethod
    def load_ndjson(path) -> list:
        # convenience function for loading 1 path
        with open(path) as fin:
            return ndjson.load(fin)

    @staticmethod
    def export_ndjson(dobj, outpath) -> None:
        # convenience function for exporting 1 object
        with open(outpath, 'w') as fout:
            ndjson.dump(dobj, fout)
    
    @staticmethod
    def load_csv(path) -> pd.DataFrame:
        # load csv and convert to a list of dicts (standard ndjson format)
        df = pd.read_csv(path, sep=';')
        return [row.to_dict() for i, row in df.iterrows()]
    
    @staticmethod
    def export_csv(dobj, outpath) -> None:
        # convert standard ndjson format to df, export as csv
        df = pd.DataFrame(dobj)
        df.to_csv(outpath, index=False, sep=';')


    def export_dobj(self, dobj, outpath) -> None:
        '''Convert object to a desired format and export it
        with the right file extension

        Parameters
        ----------
        dobj : list or dicts
            Processing output of Query()
        outpath : str
            Path for dumping the object

        Raises
        ------
        FileExistsError
            If desired path already exists

        ValueError
            If an unknown file extension is called.
        '''

        # make sure file extension is present
        if not outpath.endswith('.csv' or '.ndjson'):
            outpath = outpath + '.' + self.file_ext

        try:
            # resolve path
            if not os.path.exists(outpath):
                pass
        except:
            raise FileExistsError('{} already exists! Stoping execution.'.format(outpath))

        # get file extension
        if self.file_ext == 'ndjson':
            # export dobj as is
            self.export_ndjson(dobj, outpath)

        elif self.file_ext == 'csv':
            # convert dobj to a dataframe and export
            self.export_csv(dobj, outpath)

        else:
            # raise error if non-implemented formats are called
            raise ValueError('`export_file_ext:` Exporting only works in .csv or .ndjson formats')
        


class InfoMediaQuery(Query):
    '''
    File loop
    Folder loop
    Dataset loop
    '''
    
    # TODO look into __new__ or __init_subclass__
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)


    # INFOMEDIA PATH MANIPULATION METHODS
    @staticmethod
    def parse_path_date(path, pattern) -> datetime.datetime:
        '''
        Parse dates in filename.

        Parameters
        ----------
        path : str
            Path to a file from IM's database

        pattern : `re.Pattern`
            Pattern made using self.path_date_pattern()
        ''' 
        date_match = pattern.search(path)
        fdate = parse_datetime_as_naive(date_match.group())

        return fdate
    
    @staticmethod
    def path_date_pattern(folder_year=True) -> re.Pattern:
        '''
        Compile regex pattern for extracting dates from paths.

        True: yyyy (a folder's name) in path 
        False: yyyy-mm-dd (part of a filename.ndjson) in path

        Probably specific to infomedia file structure.
        '''
        if folder_year:
            return re.compile(r'(?<=/)\d{4}(?=/)')
        else:
            return re.compile(r'\d{4}-\d{2}-\d{2}')

    def flag_paths_matching_date(self, path_list) -> list:
        '''
        We don't need to open some files.
        e.g. if timerange is just 2019-03-05, all files with other date can be skipped

        Returns a list of bool, where True means "path is within timeframe".
        '''
        # compile regex pattern for matching dates in filename
        pattern = self.path_date_pattern(folder_year=False)
        # extract and parse dates in filenames
        date_list = [self.parse_path_date(path, pattern) for path in path_list]
        # return list of boolean within the time bounds
        return [self.dated(date, self.date_start, self.date_end) for date in date_list]
    
    def flag_articles_matching_date(self, file, field='PublishDate') -> list:
        '''
        Once we need to check dates inside the files...

        Returns a list of bool, where True means "article is within timeframe".
        '''
        date_list = [parse_datetime_as_naive(article[field]) for article in file]
        return [self.dated(date, self.date_start, self.date_end) for date in date_list]

    def get_paths_timeframe(self, path_list) -> list:
        '''
        Return a list of str (paths).
        '''
        # TODO do we really need this method? 
        flags_files = self.flag_paths_matching_date(path_list)
        return self.extract_matched(path_list, flags_files)


    # INFOMEDIA FIELD MANIPULATION METHODS
    # applicable for all dictionaries as well
    # TODO better KeyError when field is not found in the dataset
    def get_content_from_match_fields(self, file) -> list:
        '''
        Select some fields from all dicts (articles) in a file.
        All selected fields from one dict are concatenated into one string.

        Returns a list of strings. List: file level, string: article level
        '''
        return [
            '\n'.join(article[k] for k in self.fields_to_match) 
            for article in file
                ]

    def get_content_for_export(self, file):
        '''
        Select fields to export.
        '''
        return [
            {key: article[key] for key in self.limited_export}
            for article in file
            ]


    # INFOMEDIA SUBSET MAKING METHODS
    # (1) FILE LOOP
    def process_filepath(self, path, remove_duplicates=False) -> list:
        '''
        Extract matched articles from one file

        Returns a list of matched articles (output:list, output[i]:dict)
        from ONE file.

        Parameters
        ----------
        path : str
            path to file to process.
        '''
        file = self.load_ndjson(path)

        # get content from a file
        file_content = self.get_content_from_match_fields(file)

        # regex-match query
        extract_list = self.flag_matched_articles(file_content, self.query_pattern)

        if self.condition_pattern:
            # if condition is desired, regex-match condition
            condition_matched = self.flag_matched_articles(
                file_content,
                self.condition_pattern)
            # both query & condition must be True
            extract_list = self.find_mutual_trues(extract_list, condition_matched)

        # extra layer of processsing if dates are finegrained
        if self.finegrained_dates:
            dates_matched = self.flag_articles_matching_date(file)
            extract_list = self.find_mutual_trues(extract_list, dates_matched)

        # extract matched articles
        match = self.extract_matched(file, extract_list)

        # curb content if desiderd
        if self.limited_export:
            match = self.get_content_for_export(match)

        return match


    # (2) FOLDER LOOP
    # abstraction 1: EVERYTHING IN A SINGLE FILE
    def collect_subset_in_list(self, paths, progress_bar=True) -> list:
        '''
        Input paths, export all matches into a single file.

        Returns a list of matched articles (output:list, output[i]:dict)
        from MULTIPLE files

        Parameters
        ----------
        paths : list, optional
            List of filepaths to run.
            If None is specified, self.paths is used.

        progress_bar : bool, optional
            Show progress bar? By default False.
        '''

        # take paths from init if nothing is specified
        if not paths:
            paths = self.paths

        # dates are taked directly from init by the method
        paths_to_process = self.get_paths_timeframe(paths)

        # Instantiate progres bar
        if progress_bar:
            paths_to_process = tqdm(paths_to_process)

        # iterate
        all_matched = []
        for path in paths_to_process:
            file_subset = self.process_filepath(path)
            # don't add empties
            if file_subset:
                # extend because self.extract_matched() returned a list already
                all_matched.extend(file_subset)
                
        return all_matched


    # abstraction 2: EXPORT TO INDIVIDUAL FOLDER (NEWSPAPER) FILES
    def collect_in_media_folder(self, one_year_folder=False, progress_bar=True) -> None:
        '''
        Input folder paths (not filepaths).
        Returns nothing, saves files straight to self.export_dir.

        Parameters
        ----------
        one_year_folder : bool, optional
            True is when using a yearly subset {one media folder}/{all ndjson files}
            Flase is when using IM database {one media folder}/{any year}/{all ndjson files}

        progress_bar : bool, optional
            Show progress bar? By default False.
        '''

        # test if all paths are folders
        assert all([os.path.isdir(path) for path in self.paths])
        folders_to_process = self.paths

        if progress_bar:
            folders_to_process = tqdm(folders_to_process)

        # iterate over folder
        for folder in folders_to_process:

            # find out filestructure for file paths
            if one_year_folder:
                paths_within_folder = glob.glob(
                    # {one media folder}/{all ndjson files}
                    os.path.join(folder, '*.ndjson')
                    )

            else:
                paths_within_folder = glob.glob(
                    # {one media folder}/{any year}/{all ndjson files}
                    os.path.join(folder, '*', '*.ndjson')
                    )

            # time filtering
            paths_to_process = self.get_paths_timeframe(
                paths_within_folder
            )

            # collect matched into one output
            media_folder_subset = self.collect_subset_in_list(
                paths_to_process,
                # turn progress bar off
                progress_bar=False
                )

            # don't export empties!
            if media_folder_subset:
                # filename for folder subset
                fname = self.basepath_to_name(folder, ext=self.file_ext)
                outpath = os.path.join(self.export_dir, fname)

                # try to export
                # will fail if ext is not right
                self.export_dobj(media_folder_subset, outpath)


    def remove_duplicates(self, export_dir, clear_html=False):
        # BAREBONES, UNSMOOTH

        # find all files generated
        file_pattern = os.path.join(self.export_dir, '*.' + self.file_ext)
        exported_paths = glob.glob(file_pattern)

        # regex patterns for html
        if clear_html:
            pat_html = RegxFilter(pattern=r"<.*?>") # remove html tags
            pat_ws = RegxFilter(pattern=r" +") # remove extra spacing to deal with p1 header

        print('[info] Removing duplicates')
        for path in tqdm(exported_paths):
            # load correctly
            # TODO GENERIC LOADER FUNCTION
            if self.file_ext == 'csv':
                file = self.load_csv(path)
            elif self.file_ext == 'ndjson':
                file = self.load_ndjson(path)

            if file:
                # don't work on empties
                body_text = [article['BodyText'] for article in file]

                # get ids of duplicates
                ids_duplicates = flag_fuzzy_duplicates(
                    body_text,
                )

                # flip list to get ids to keep
                ids_non_duplicates = [not el for el in ids_duplicates]

                # keep only non-duplicates
                match = self.extract_matched(file, ids_non_duplicates)

                if clear_html:
                    # TODO
                    # clear html belongs in tekisuto!
                    def remove_html(dict, key):
                        text = dict[key]
                        text = pat_html.preprocess(text)
                        text = pat_ws.preprocess(text)
                        return text
                    
                    print('[info] removing html')
                    for article in match:
                        for key in ['Heading', 'SubHeading', 'Paragraph', 'BodyText']:
                            article[key] = remove_html(article, key)

                # export
                fname = os.path.basename(path)
                self.export_dobj(match, os.path.join(export_dir, fname))

            else:
                pass
