# -*- coding: utf-8 -*-

import os
import errno
import argparse
from src.catalog import Catalog


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(errno.ENOENT, os.strerror(errno.ENOENT), string)


def file_path(string):
    if os.path.isfile(string):
        return string
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), string)


def main():
    parser = argparse.ArgumentParser(prog='deep-search.py',
                                     usage='The following options are acceptable:\n'
                                           '* Index the catalog:  '
                                           'python %(prog)s index --catalog_path path-to-catalog\n'
                                           '* Search in the catalog:  '
                                           'python %(prog)s search --catalog_path path-to-catalog '
                                           '--results_path path-to-results --text text1 text2 textN\n',
                                     description='Search text or image in your catalog.')

    parser.add_argument('mode', help='Need to provide an action: search or index.', choices=['search', 'index'])
    parser.add_argument('--catalog_path', help='Path to the catalog directory.', type=dir_path, required=True)
    args, leftovers = parser.parse_known_args()

    catalog = Catalog(args.catalog_path)

    if args.mode == 'index':
        catalog.index()
    else:
        parser.add_argument('--results_path', help='Path to the catalog directory.', type=dir_path, required=True)
        parser.add_argument('--text', nargs='+', required=True)
        args = vars(parser.parse_args())
        catalog.search_text(args['text'])

        print(args['text'])


if __name__ == "__main__":
    main()
