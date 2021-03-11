'''
Extract subset from one raw news-file.

If your data is in a single folder:
- and you want to save subset in a single file
- and you want to keep the original file structure

If your data is in multiple folders:
- and you want to save subset in a single file
- and you want to save subset in one file per folder
- and you want to keep the original file structure
'''
# %%

import os
import re
import glob
import datetime
import warnings

import ndjson
import numpy as np
from tqdm import tqdm
from mdutils.mdutils import MdUtils
from ciso8601 import parse_datetime_as_naive, parse_datetime
# probalby need to swtich to parse_datetime once UK/US newspapers start coming in

# %%
class Query:
    '''
    Set everything up.
    '''
    def __init__(self,
                query_pattern,
                date_start,
                date_end,
                condition_pattern=None,
                subset_tag=None,
                export_parent_dir=None,
                filefinding_pattern=None,
                folderfinding_pattern=None,
                fields_to_match=None,
                limited_export=False) -> None:
        
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
        # making sure they are sorted
        if filefinding_pattern:
            # TODO resolve?
            self.paths = sorted(glob.glob(filefinding_pattern))
            self.paths_to_process = self.get_paths_timeframe(self.paths)

        if folderfinding_pattern:
            # TODO validate only folders are in the mix
            self.folders = sorted(glob.glob(folderfinding_pattern))


        # exportation
        # TODO
        # keep track of desired datatype
        # and make it possible to do multiple datatypes?!

        # output paths
        if export_parent_dir:
            self.parent_dir = self.resolve_path(export_parent_dir)
            if subset_tag:
                # if tag was provided, make sure it's a string, then save
                self.subset_tag = self.validate_string(subset_tag)
            self.subset_dir = self.name_subset_dir()
            # create output directory
            self.create_folder(self.subset_dir)
            self.subset_data_dir = os.path.join(self.subset_dir, 'subset_data')
            self.create_folder(self.subset_data_dir)


    # INPUT VALIDATION
    def validate_filefinding_paths(self, path_list) -> list:
        '''
        Make sure paths exist and are a list (iterable).
        '''
        # FIXME unsmooth checking for empty
        if not path_list:
            raise ValueError('No files to process found! Check path patterns in input.')

        if isinstance(path_list, str):
            path_list = [path_list]
        
        path_list = [self.resolve_path(path) for path in path_list]

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
    def validate_match_fields(fields_to_match) -> tuple:
        '''
        Turns inputed match_fields into a tuple for faster runing.

        TODO 
        - correct another datatypes?
        - check if they are in the dataset
            - might have to load a sample of the dataset to check
        '''
        if isinstance(fields_to_match, list):
            fields_to_match = tuple(fields_to_match)
        
        return fields_to_match

    # REGEX
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

    # PATH MANIPULATION
    @staticmethod
    def validate_string(string) -> str:
        if not isinstance(string, str):
            string = str(string)
        
        return string

    @staticmethod
    def resolve_path(path) -> str:
        if os.path.exists(path):
            return path
        else:
            raise FileNotFoundError('path not found `{}`'.format(path))

    @staticmethod
    def folder_to_fname(folder_path, ext='ndjson') -> str:
        '''
        From a path to folder, get the name of the deepest folder.
        If `ext` is specified, add a file extension to the name.  

        TODO 
        - ext should be T/F and get particulat file extension from export_datatype!
        '''
        if ext:
            return os.path.basename(os.path.normpath(folder_path)) + '.' + ext
        else:
            return os.path.basename(os.path.normpath(folder_path)) 

    def name_subset_dir(self) -> str:
        # add timestamp to outfolder
        yymmddhhmm = datetime.datetime.now().strftime('%y%m%d%H%M')

        # subset flags
        # could be that here processing flags could be automatically detected

        folder_name = yymmddhhmm + '_' + self.subset_tag
        outpath = os.path.join(self.parent_dir, folder_name)

        return outpath

    @staticmethod
    def create_folder(fpath) -> None:
        '''
        TODO
        - confirm saving path?
        '''
        # check if folder already exists and prevent overwriting!
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        else:
            raise FileExistsError('path already exists {}'.format(fpath))


    # EXTRACTION METHODS
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
        return [x for i, x in enumerate(base_list) if flag_list[i]]

    @staticmethod
    def find_mutual_trues(list1, list2) -> np.ndarray:
        '''
        Flag indices where BOTH lists have Ture

        Parameters
        ----------
        list1, list2 : bool
            list of bool of the same length
        '''
        assert len(list1) == len(list2)

        if list1 and list2:
            list1 = np.array(list1)
            list2 = np.array(list2)

            return list1 & list2

        else:
            return None

    # FILTERING
    ## DATE based
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
        # TODO this where I wanted to check for sorted paths 
        # select folders
        flags_files = self.flag_paths_matching_date(
            path_list
            )

        return self.extract_matched(path_list, flags_files)

    ## SOURCE based
    # TODO
    # don't open some folders, because it's a newspaper we don't want.

    ## CONTENT based
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


    # I/O
    ## IMPORT
    @staticmethod
    def load_file(path) -> list:
        '''
        Imports ndjson file corresponding to one day of content from one source.

        TODO 
        benchmark compare with generators loading (ndjson.reader)
        '''
        with open(path) as fin:
            file = ndjson.load(fin)

        return file

    ## EXPORT
    @staticmethod
    def export_ndjson(dobj, outpath) -> None:
        '''
        '''
        # TODO benchmark ndjson.reader
        with open(outpath, 'w') as fout:
            ndjson.dump(dobj, fout)

    @staticmethod
    def export_csv(dobj, outpath) -> None:
        '''
        '''
        pass

    # README
    def create_readme(self, maximal=False) -> None:
        '''Generates a markdown readme file.
        Saved into the subset directory.

        Parameters
        ----------
        maximal : bool
            If maximal readme is desired,
            articles will be counted.
        '''
        if self.subset_tag:
            subset_tag = self.subset_tag
        else:
            subset_tag = 'subset'

        # create
        readme = MdUtils(
            file_name= os.path.join(self.subset_dir, 'readme_' + subset_tag),
            title=subset_tag
            )

        # produce content
        date_generated = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        str_date_start = self.date_start.strftime('%Y-%m-%d %H:%M:%S')
        str_date_end = self.date_end.strftime('%Y-%m-%d %H:%M:%S')

        # SECTION 1: DATE GENERATED
        readme.new_header(level=1, title='Date', style='setext')
        readme.new_paragraph(date_generated)

        # SECTION 2: SUBSET DESCRIPTION
        readme.new_header(level=1, title='Description', style='setext')
        # start date, end date
        readme.new_line(
            'Articles from {} to {}'
            .format(str_date_start, str_date_end)
        )

        # regex patterns for query
        readme.new_line(
            'Regex pattern of *query*: ``{}``'
            .format(self.query_pattern)
            )
        # regex pattern for condition
        if self.condition_pattern:
            readme.new_line(
                'Regex pattern of *condition*: ``{}``'
                .format(self.condition_pattern)
                )

        # SECTION 3: SOURCE
        # # TODO
        # if self.filefinding_pattern:
        #     pass

        # if self.folderfinding_pattern:
        #     pass

        if maximal:
            pass
            # generate tally content
            # TODO - maybe too complicated?

            # table_contet = [
            #     'Number of Days', ''
            #     'Number of Articles', '',
            #     'Number of Media Houses', '',
            #     'Avg Articles per Media House', ''
            # ]

            # SECTION 4: TABLE OF DETAILS
            # readme.new_header(level=1, title='Setext Header 1', style='setext')
            # readme.new_table(columns=2, rows=4, text=table_contet, text_align='left')

        readme.create_md_file()


class InfoMediaQuery(Query):
    '''
    File loop
    Folder loop
    Dataset loop
    '''
    
    # TODO look into __new__ or __init_subclass__
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    # (1) FILE LOOP
    def process_filepath(self, path) -> list:
        '''
        Extract matched articles from one file

        Returns a list of matched articles (output:list, output[i]:dict)
        from ONE file.

        Parameters
        ----------
        path : str
            path to file to process.
        '''
        # import ndjson
        # TODO
        # check for empty files?
        file = self.load_file(path)

        # get content from a file
        file_content = self.get_content_from_match_fields(file)

        # regex-match query
        extract_list = self.flag_matched_articles(file_content, self.query_pattern)

        if self.condition_pattern:
            # if condition is desired, regex-match condition
            condition_matched = self.flag_matched_articles(
                file_content,
                self.condition_pattern
                )
            # both query & condition must be True
            # FIXME delete test cases
            self.test_extract_list = extract_list
            self.test_condition_matched = condition_matched
            extract_list = self.find_mutual_trues(extract_list, condition_matched)

        # extra layer of processsing if dates are finegrained
        # FIXME untested
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
    def collect_subset_in_list(self, paths_to_process, progress_bar=False) -> list:
        '''
        Input paths, export all matches into a single file.

        Returns a list of matched articles (output:list, output[i]:dict)
        from MULTIPLE files

        Parameters
        ----------
        paths_to_process : str|list
            List of filepaths to run
        '''
        # # TODO 
        # # dates are taked directly from init by the method
        # paths_to_process = self.get_paths_timeframe(
        #     self.paths
        # )

        # TODO
        # - paths_to_process should be the point where we paralelize.
        #     - I was hoping they would be initialized somewhere else..

        paths_to_process = self.validate_filefinding_paths(paths_to_process)

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
    def collect_in_media_folder(self,
        folders_to_process, one_year_folder=False, progress_bar=False) -> None:
        '''
        Input folder paths (not filepaths).

        Returns nothing, saves files straight to self.subset_data_dir.

        Parameters
        ----------
        folders_to_process : str|list
            List of directories to go into and collect
        
        one_year_folder : bool
            True is when using a yearly subset {one media folder}/{all ndjson files}
            Flase is when using IM database {one media folder}/{any year}/{all ndjson files}
        '''
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
                fname = self.folder_to_fname(folder)
                outpath = os.path.join(self.subset_data_dir, fname)

                # resolve outpath & export
                if not os.path.exists(outpath):
                    self.export_ndjson(media_folder_subset, outpath=outpath)
                
                else:
                    warnings.warn('{} already exists! Overwriting content!'.format(outpath))



    # # abstraction 3: EXPORT TO ORIGINAL FILE STRUCTURE
    # def collect_in_original_filestructure(paths_to_process, fields_to_match, query_pattern, condition_pattern):
    #     '''
    #     Input paths, export all matches into files with the same 
    #     resolution as IM database

    #     Parameters
    #     ----------
    #     paths_to_process : str|list
    #         List of filepaths to run
    #     '''
    #     for path in tqdm(paths_to_process):
    #         file_subset = process_filepath(
    #             path,
    #             fields_to_match,
    #             query_pattern,
    #             condition_pattern
    #             )
            
    #         export_ndjson(file_subset, outpath=None)

