# -*- coding: utf-8 -*-
import glob
import logging
import os
import re
import shutil
from shutil import copytree, ignore_patterns
from typing import Tuple, List

import pandas as pd


def get_id(id_raw: str, base_id: int, image_to_id: dict, base_path: str = "") -> Tuple[str, int]:
    """
    Takes the content of the ID column and extracts the ID.
    If there is no ID, it assigns a new one.
    :param id_raw: Content of ID column in the xls
    :param base_id: ID to assign if no proper value can be extracted. Typically for 'no match' cases
    :param image_to_id: Dict that maps each image to its id
    :param base_path:
    :return: extracted_id, new_base_id
    """
    base_path = base_path[:-4]  # remove Year from base_path
    id_raw = id_raw.strip()

    # Some whales have nicknames instead of an ID
    # those names can be also in combination with other words
    nick_names = ['nike', 'whitehead', 'cf_of_whitehead', 'cf_of_fingers']
    nick_names_exact = ['19']
    for name in nick_names:
        matches = re.findall(f'^{name}', id_raw.lower())
        if len(matches) > 0:
            return matches[0], base_id
    for name in nick_names_exact:
        if id_raw == name:
            return name, base_id

    # A -1 indicates that the photo was not good enough quality for matching. Bad angle or only half a tail.
    bad_pic = ['bad pic', 'poor pic', 'no flukes', 'ck', 'chk', 'smooth']
    # Its possible that there is a -1 somewhere in the id_raw so the -1 is only valid at the start or end
    if (id_raw.lower() in bad_pic) or len(re.findall('^-1|-1$', id_raw)) > 0:
        return '-1', base_id

    # There are some images with a different naming schema PM-HC-13Sept12-SM-0204 that we dont have in our dataset
    if id_raw.startswith('HC-') or id_raw.startswith('PM-HC-'):
        return str(base_id), base_id + 1

    # Whales that do not have an ID yet are identified by a special string
    # In some occasions, there is no ID string at all
    # new ID can be also combined with other indicators (cf)
    no_id_yet = ['no match', 'nan', 'new id', 'new']
    for nid in no_id_yet:
        matches = re.findall(nid, id_raw.lower())
        if len(matches) > 0:
            return str(base_id), base_id + 1

    # Some whales have IDs from other Organisations, these will be replaced with new IDs
    # i.e. PM-ET-2012-PM-PM-0804, Dop-(1999)
    organizations = ['et', 'espaco tal', 'fut', 'sao miguel', 'ta', 'terra azul', 'naf',  'noaa', 'rui', 'poss', 'dop']
    for org in organizations:
        matches = re.findall(org+'(.*(199[0-9]|20[0-2][0-9])|-pm|-jq)', id_raw.lower())
        if len(matches) > 0:
            return str(base_id), base_id + 1

    # If a whale does not have an ID yet it can also be identified by an earlier image
    # Filenames can have additional notes (i.e. qua X), so we cut the raw id at the point where the filename should end.
    # Sometimes a shorter version of the image name is used
    # Sometimes they use only the Date without PM-WWA in the beginning
    matches = re.findall('[0-9]{8}-[A-Za-z0-9]{3,6}', id_raw)
    if len(matches) >= 1:
        id_path = str(os.path.join(base_path + matches[0][:4], 'PM-WWA-' + matches[0] + '.jpg'))
        try:
            return image_to_id[id_path], base_id
        except KeyError:  # If it cant find the ref Key (non parsed Folders, img older 2005, ...) create a new ID
            logging.info(f'Missing reference to previous image: {id_raw} => {id_path}')
            return str(base_id), base_id + 1

    # There are raw_ids with other year formats YYYY-DDMM-NR (prob ids older 2005)
    # The Image Nr can also include letters from a-Z
    matches = re.findall('[0-9]{4}-[0-9]{4}-[A-Za-z0-9]{3,6}', id_raw)
    if len(matches) >= 1:
            fid = matches[0][:4] + matches[0][7:9] + matches[0][5:7] + matches[0][9:]
            id_path = str(os.path.join(base_path + matches[0][:4], 'PM-WWA-' + fid + '.jpg'))
            try:
                return image_to_id[id_path], base_id
            except KeyError:  # If it cant find the ref Key (non parsed Folders, img older 2005, ...) create a new ID
                logging.info(f'Missing reference: {id_raw} => {id_path}')
                return str(base_id), base_id + 1

    # Images (PM-WWA- or WWA-) with an unknown date format (i.e. YYYY-MM-DD, YY-MM-DD, YYYY-MMM-DD, DDMMYYYY, DD/MM/YYYY) will be filtered and get an new id
    matches = re.findall('(^(19|20)[0-9]{2}.?[A-Za-z0-9]{2,3}.?[0-9]{2}|^[0-9]{2}.?[A-Za-z0-9]{2,3}.?(19|20)[0-9]{2})', id_raw)
    if(id_raw.lower().startswith('wwa-') or id_raw.lower().startswith('pm-wwa-') or len(matches) >= 1):
        return str(base_id), base_id + 1

    # Known whales have a 2 to 4-digit ID
    # Nicknames can also be combines with IDs, e.g. Nike (2628)
    matches = re.findall('[0-9]{2,4}', id_raw)
    if len(matches) == 1:
        # check that its not part of '... cf of *match*', note: '*match* cf of ....' is an valid id
        if not re.match(f'(calf|cf).*{matches[0]}', id_raw):
            return matches[0], base_id
    # Several IDs can indicate calves, e.g. '3094 cf of 1645'
    # Also, ID identification can be in the ID col, e.g. 2937 = 3418
    elif len(matches) > 1 and (('cf of' in id_raw.lower()) or ('calf of' in id_raw.lower())
            or ('=' in id_raw) or ('see' in id_raw) or ('from' in id_raw)):
        return matches[0], base_id
    # We also can have duplicate IDs, e.g. 2402 (was 3059)
    elif 'was' in id_raw and len(matches) > 0:
        return matches[0], base_id
    # When there are multiple id's and non of the previous key words in the col, use the first one
    elif len(matches) >= 2:
        return matches[0], base_id

    # some calf's are only marked as "cf" or "cf of ..." with no own id
    if('cf' in id_raw.lower()) or ('calf' in id_raw.lower()):
        return str(base_id), base_id + 1

    msg = f"Unknown ID: {id_raw}"
    raise ValueError(msg)


def get_row_images(images_raw: list, input_data_path: str) -> Tuple[List, List]:
    """
    Given the image part of a row in the xls, returns all images that were found
    in the corresponding folder and all images that weren't found
    :param images_raw: List of images to look for
    :param input_data_path:  Location where to look for the images
    :return: found_images, missing_image
    """
    search_results = [get_image(image_raw, input_data_path) for image_raw in images_raw
                      if not pd.isna(image_raw)]
    found_images = [image for sublist in search_results for image in sublist]
    missing_images = [image_raw for sublist, image_raw in zip(search_results, images_raw)
                      if len(sublist) == 0 and not pd.isna(image_raw)]
    return found_images, missing_images


def get_image(image_raw: str, input_data_path: str) -> List[str]:
    """
    Checks if the folder input_data_path contains a file that fits to image_raw.
    :param image_raw: Image name in the excel file
    :param input_data_path: Where to look for the the image
    :return: List with the file names if a matching file was found, otherwise an empty string
    """
    logging.debug(f"Parsing {image_raw}")
    file_types = ['.jpg', '.tif']
    image_raw = str(image_raw).strip()  # basic cleansing

    # Some images are missing the PM-WWA prefix
    # e.g. 20060612-071
    if image_raw.startswith('20'):
        parts = image_raw.split('-')
        if len(parts) == 2 and len(parts[0]) == 8:
            image_raw = 'PM-WWA-' + image_raw

    # Sometimes the M of the PM prefix is missing
    if image_raw.startswith('P-'):
        image_raw = 'PM' + image_raw[1:]

    # Some images are missing the PM-WWA prefix
    # e.g. WWA-20050511-012
    if image_raw.startswith('WWA'):
        parts = image_raw.split('-')
        if len(parts) == 3:
            image_raw = 'PM-' + image_raw

    # Sometimes there are comments after the actual image name, e.g.
    # PM-WWA-20050707-271 X
    if image_raw.startswith('PM-WWA') and len(image_raw.split(' ')) > 1:
        image_raw = image_raw.split(' ')[0]

    # There are sometimes typos in the date or suffix of the filename
    if image_raw.startswith('PM-WWA-2'):
        parts = image_raw.split('-')

        date = parts[2]

        # PM-WWA-200504130003b -> PM-WWA-20050413-003b.jpg
        if len(date) > 8 and len(parts) == 3:
            parts.append(date[9:])
            date = date[:8]
        assert date.isdigit()
        # PM-WWA-22060422-25 -> PM-WWA-20060422-025.JPG
        if date.startswith('22'):
            date = '20' + date[2:]
        # PM-WWA-200600507-003 -> PM-WWA-20060507-003.jpg
        if date[4:6] == '00':
            date = date[:4] + date[5:]
        parts[2] = date

        suffix = parts[3]
        if len(suffix) > 1 and not suffix[0].isdigit():
            pre = suffix[0]
            suffix = suffix[1:]
        else:
            pre = ''

        # Missing padding in the end, e.g.  PM-WWA-20060601-2
        if suffix.isdigit() and len(suffix) < 3:
            suffix = suffix.rjust(3, '0')

        # PM-WWA-20060610-006/007 two images in one
        # PM-WWA-20060531-E021/22
        if '/' in suffix:
            img_suffixes = suffix.split('/')
            image_raws = ['-'.join(parts[:3] + [pre + s]) for s in img_suffixes]
            return get_row_images(image_raws, input_data_path)[0]

        parts[3] = pre + suffix
        image_raw = '-'.join(parts)

    # Base case: Just the ending is missing
    path_candidates = [f"{image_raw}{end}" for end in file_types]
    for p in path_candidates:
        img_path = os.path.join(input_data_path, p)
        if os.path.exists(img_path):
            return [img_path]

    # Some tif files has a secondary suffix, e.g.
    # PM-WWA-20060615-A130-01.tif
    path_candidates = [f"{image_raw}-01{end}" for end in file_types]
    for p in path_candidates:
        img_path = os.path.join(input_data_path, p)
        if os.path.exists(img_path):
            return [img_path]

    # In 2006 there are some examples where a digit is missing. e.g.
    # PM-WWA-20060420-A41 -> PM-WWA-20060420-A041.jpg
    if image_raw[-2:].isdigit() and image_raw.startswith('PM-WWA-200'):
        path_candidates = [f"{image_raw[:-2]}0{image_raw[-2:]}{end}" for end in file_types]
        for p in path_candidates:
            img_path = os.path.join(input_data_path, p)
            if os.path.exists(img_path):
                return [img_path]
    return []


def parse_excel(input_data_path: str, id_to_images=None, image_to_id=None, base_id: int = 5000) \
        -> Tuple[dict, dict, set, int]:
    """
    Looks for and reads the xls file in the location input_data_path
    Creates a two dictionaries, one that maps whale IDs to all images that show this
    whale, and another one that maps each image that exists on the path to its
    corresponding whale ID. Additionally, keeps track of all IDs to match (as a set of tupels)

    Note that each line in the excel file is a new individual. Not every individual
    already has an assigned ID.
    The id -1 indicates that the pic is not usable
    :param input_data_path:
    :param id_to_images: Dictionary that maps whale IDs to a list of all images for the respective whale
    :param image_to_id: Dictionary that maps images to the corresponding whale id
    :param base_id: ID to use for new matches
    :return: id_to_images, image_to_id, ids_to_match, base_id
    """
    if image_to_id is None:
        image_to_id = {}
    if id_to_images is None:
        id_to_images = {}

    # Find the xlsx and get the id - image connections
    xls_files = glob.glob(input_data_path + '/*.xls*')
    xls_files = [f for f in xls_files if not os.path.basename(f).startswith('~$')]  # Remove tmp files
    assert len(xls_files) == 1
    xls = pd.read_excel(xls_files[0], header=None)
    logging.info(f"Parsing {xls_files[0]}")
    ids_to_merge = set()

    for index, row in xls.iterrows():
        id_raw = row[0]
        # There are some empty rows that we can ignore
        if pd.isna(id_raw):
            continue
        # 2006 has a header row that we need to ignore
        if id_raw == 'ID ( X=Done in ID Table)':
            continue
        # 2011 summary row that we need to ignore
        if id_raw == 'END BIOSPHERE' or id_raw == 'END OF BIOSPHERE':
            continue

        id_raw = str(id_raw).strip()

        whale_id, base_id = get_id(id_raw, base_id, image_to_id, input_data_path)
        assert isinstance(whale_id, str)
        found_images, missing_images = get_row_images(row[1:], input_data_path)
        if len(found_images) == 0:
            # Missing files
            if whale_id != '-1':
                logging.error(f"No image at row {index} for ID {whale_id}, {missing_images}")
            continue
        # Check for year error
        #if re.match('(20[0-9]{2}|199[0-9])', whale_id):
        #    logging.warning(f'Check {whale_id} from {id_raw}')
        # Assign images to the ID
        try:
            id_to_images[whale_id] = id_to_images[whale_id].union(found_images)
        except KeyError:
            id_to_images[whale_id] = set(found_images)

        # Assign the ID to the images
        for img in found_images:
            if img not in image_to_id:
                image_to_id[img] = whale_id
            elif image_to_id[img] != whale_id and whale_id == '-1':
                # unlink an image from a whale_id when it is marked somewhere else as bad
                id_to_images[image_to_id[img]].remove(img)
                del image_to_id[img]
            elif image_to_id[img] != whale_id:
                # Keep track of all IDs to merge
                other_id = image_to_id[img]
                assert whale_id != '-1' and other_id != '-1', f"{whale_id}, {other_id}, {img}"
                ids_to_merge.add((min(whale_id, other_id), max(whale_id, other_id)))
                whale_id = min(whale_id, other_id)
    return id_to_images, image_to_id, ids_to_merge, base_id


def merge_ids_manual(id_to_images, image_to_id):
    """
    Contains fixes for all the DB errors that were found after the start of the challenge by Lisa.
    :param id_to_images:
    :param image_to_id:
    :return:
    """
    id_to_images, image_to_id = ordered_merge('2628', 'nike', id_to_images, image_to_id)
    id_to_images, image_to_id = ordered_merge('2929', '2620', id_to_images, image_to_id)
    id_to_images, image_to_id = ordered_merge('1554', '2710', id_to_images, image_to_id)
    id_to_images, image_to_id = ordered_merge('3783', '3848', id_to_images, image_to_id)
    id_to_images, image_to_id = ordered_merge('2907', '3172', id_to_images, image_to_id)
    id_to_images, image_to_id = ordered_merge('2556', '2934', id_to_images, image_to_id)
    id_to_images, image_to_id = ordered_merge('2351', '3414', id_to_images, image_to_id)
    id_to_images, image_to_id = ordered_merge('2935', '2936', id_to_images, image_to_id)
    id_to_images, image_to_id = ordered_merge('2808', '3347', id_to_images, image_to_id)
    id_to_images, image_to_id = ordered_merge('1811', '3033', id_to_images, image_to_id)
    id_to_images, image_to_id = ordered_merge('3263', '3970', id_to_images, image_to_id)
    id_to_images, image_to_id = ordered_merge('3973', '5373', id_to_images, image_to_id)
    id_to_images, image_to_id = ordered_merge('2776', '3177', id_to_images, image_to_id)
    id_to_images, image_to_id = ordered_merge('2233', '2223', id_to_images, image_to_id)
    id_to_images, image_to_id = ordered_merge('3014', '3213', id_to_images, image_to_id)
    return id_to_images, image_to_id


def ordered_merge(id_to_keep: str, id_to_delete: str, id_to_images: dict, image_to_id: dict) -> Tuple[dict, dict]:
    try:
        logging.warning(f"ID {id_to_delete}:{id_to_images[id_to_delete]} is merged into ID {id_to_keep}:{id_to_images[id_to_keep]}")
        id_to_images[id_to_keep] = id_to_images[id_to_keep].union(id_to_images[id_to_delete])
        for img in id_to_images[id_to_delete]:
            image_to_id[img] = id_to_keep
        del id_to_images[id_to_delete]
    except KeyError:
        logging.error('Could not merge %s and %s' % (id_to_keep, id_to_delete))
    return id_to_images, image_to_id


def merge_ids(id_1: str, id_2: str, id_to_images: dict, image_to_id: dict) -> Tuple[dict, dict]:
    logging.warning(f"Merging IDs {id_1}:{id_to_images[id_1]}, {id_2}:{id_to_images[id_2]}")
    min_id = min(id_1, id_2)
    max_id = max(id_1, id_2)
    id_to_images, image_to_id = ordered_merge(min_id, max_id, id_to_images, image_to_id)
    return id_to_images, image_to_id


def copy_images(id_to_images: dict, out_data_path: str) -> None:
    """
    Takes a dict that maps IDs to filesnames and an output folder.
    Creates for each ID a new subfolder in out_data_path and copies all images that belong to this ID into
    the subfolder.
    :param id_to_images: A dictionary that maps whale IDs to a list of images.
    :param out_data_path: The location where the images will be copied to
    :return:
    """
    # Create folders
    for whale_id in id_to_images:
        out_dir = os.path.join(out_data_path, str(whale_id))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        # Move the files according to the id, report all images that have no match
        for img in id_to_images[whale_id]:
            shutil.copy(img, out_dir)
    return


def sort_yearly_folder(input_data_path: str, id_to_images=None, image_to_id=None, base_id:int = 5000) \
        -> Tuple[dict, dict, int]:
    """
    All pictures that were done in one year are stored in one folder, together
    with an excel that maps pictures to IDss. This function reads the excel, extracts
    the IDs and groups all images by their IDS so that the folder out_data_path contains
    one folder per ID and all images of that ID are copied to the folder.

    The functions reports all images that it cannot map to IDs and all IDs for which there
    are no images.

    :param input_data_path: A folder containing images and a xls file mapping images to IDs
    :param id_to_images: Dictionary that maps whale IDs to a list of all images for the respective whale
    :param image_to_id: Dictionary that maps images to the corresponding whale id
    :return:
    """
    if image_to_id is None:
        image_to_id = {}
    if id_to_images is None:
        id_to_images = {}
    logging.info(f"Sorting {input_data_path}")
    # Read all ID-IMG mappings from the excel
    id_to_images, image_to_id, ids_to_merge, base_id = parse_excel(input_data_path,
                                                                   id_to_images,
                                                                   image_to_id, base_id)

    # Merge ids that share the same image
    for id_1, id_2 in ids_to_merge:
        id_to_images, image_to_id = merge_ids(id_1, id_2, id_to_images, image_to_id)

    # Check that every img file has a corresponding id
    image_files = glob.glob(input_data_path + '/*.jpg*') + glob.glob(input_data_path + '/*.tif*')
    missing_imgs = 0
    for img in image_files:
        # Force lower case for the ending
        parts = img.split('.')
        parts[-1] = parts[-1].lower()
        img = '.'.join(parts)
        try:
            image_to_id[img]
        except KeyError:
            missing_imgs += 1
            logging.warning(f"Missing ID for {img}")
    logging.info(f'{missing_imgs} images have no ID.')

    return id_to_images, image_to_id, base_id


def sort_by_id(raw_data_path: str = None, out_data_path: str = None):
    """
    Expects raw_data_path to contain one folder per year. Every folder contains an xls file
    and images, with the xls file defining the mapping between whale ids and images.
    Parses all folders and sorts creates a new folder in out_data_path for each whale id
    and copies all images for an id in the newly created folder.
    :param raw_data_path:
    :param out_data_path:
    :return:
    """
    if raw_data_path is None:
        raw_data_path = os.path.join('data', 'cleaned_manual')
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError('Raw data dir does not exist')
    if out_data_path is None:
        out_data_path = os.path.join('data', 'sorted')

    id_to_images = {}
    image_to_id = {}
    base_id = 5000
    for folder in os.listdir(raw_data_path):
        input_data_path = os.path.join(raw_data_path, folder)
        id_to_images, image_to_id, base_id = sort_yearly_folder(input_data_path,
                                                                id_to_images,
                                                                image_to_id,
                                                                base_id)

    copy_images(id_to_images, out_data_path)
    return


def create_csv_for_evaluation(source: str, id_to_images: dict, image_to_id: dict, outfile: str = 'labels.csv'):
    """
    Creates a files that for each images lists all images that are assigned to the same whale.
    :param source:
    :param id_to_images:
    :param image_to_id:
    :param outfile:
    :return:
    """
    with open(outfile, 'w') as f:
        if (source is None) or (source == ''):
            image_files = image_to_id.keys()
        else:
            image_files = glob.glob(source + '/*.jpg*') + glob.glob(source + '/*.tif*')
        for image in image_files:
            id = image_to_id[image]
            if id == '-1':
                continue
            other_imgs = id_to_images[id] - set([image])
            cleaned_img = os.path.basename(image)
            cleaned_imgs = ','.join([os.path.basename(i) for i in other_imgs])
            f.write(f"{cleaned_img},{cleaned_imgs}\n")
    return


def create_train_test_val_data(raw_data_path: str):
    """
    Creates the training data used for the competition.
    :return:
    """
    # Training is done on all the years between 2005 and 2016
    prev_years = ['2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012',
                  '2013', '2014', '2015', '2016']
    id_to_images = {}
    image_to_id = {}
    base_id = 5000
    for i, y in enumerate(prev_years):
        source = os.path.join(raw_data_path, f'Whale Watch Azores {y}')
        id_to_images, image_to_id, base_id = sort_yearly_folder(source, id_to_images, image_to_id, base_id)
    id_to_images, image_to_id = merge_ids_manual(id_to_images, image_to_id)
    copy_images(id_to_images, os.path.join('data', 'train'))
    create_csv_for_evaluation(None, id_to_images, image_to_id, 'train.csv')

    # Copy 2017 files to test
    source_2017 = os.path.join(raw_data_path, 'Whale Watch Azores 2017')
    destination = os.path.join('data', 'test')
    copytree(source_2017, destination, ignore=ignore_patterns('*.xlsx'))

    # Copy 2018 files to val
    source_2018 = os.path.join(raw_data_path, 'Whale Watch Azores 2018')
    destination = os.path.join('data', 'val')
    copytree(source_2018, destination, ignore=ignore_patterns('*.xlsx'))

    # Test is done on 2017, validation on 2018
    id_to_images, image_to_id, base_id = sort_yearly_folder(source_2017, id_to_images, image_to_id, base_id)
    id_to_images, image_to_id, base_id = sort_yearly_folder(source_2018, id_to_images, image_to_id, base_id)
    id_to_images, image_to_id = merge_ids_manual(id_to_images, image_to_id)
    create_csv_for_evaluation(source_2017, id_to_images, image_to_id, 'test.csv')
    create_csv_for_evaluation(source_2018, id_to_images, image_to_id, 'validation.csv')
    return


if __name__ == '__main__':
    raw_data_path = os.path.join('data', 'cleaned_manual')
    out_data_path = os.path.join('data', 'processed')
    sort_by_id(raw_data_path, out_data_path)
    #create_train_test_val_data()
