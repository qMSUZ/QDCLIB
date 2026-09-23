#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#/***************************************************************************
# *   Copyright (C) 2022 -- 2026 by Marek Sawerwain                         *
# *                                  <M.Sawerwain@gmail.com>                *
# *                                  <M.Sawerwain@issi.uz.zgora.pl>         *
# *                                                                         *
# *                              by Joanna Wiśniewska                       *
# *                                  <Joanna.Wisniewska@wat.edu.pl>         *
# *                                                                         *
# *   Part of the Quantum Distance Classifier:                              *
# *         https://github.com/qMSUZ/QDCLIB                                 *
# *                                                                         *
# *   Licensed under the EUPL-1.2-or-later, see LICENSE file.               *
# *                                                                         *
# *   Licensed under the EUPL, Version 1.2 or - as soon they will be        *
# *   approved by the European Commission - subsequent versions of the      *
# *   EUPL (the "Licence");                                                 *
# *                                                                         *
# *   You may not use this work except in compliance with the Licence.      *
# *   You may obtain a copy of the Licence at:                              *
# *                                                                         *
# *   https://joinup.ec.europa.eu/software/page/eupl                        *
# *                                                                         *
# *   Unless required by applicable law or agreed to in writing,            *
# *   software distributed under the Licence is distributed on an           *
# *   "AS IS" basis, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,          *
# *   either express or implied. See the Licence for the specific           *
# *   language governing permissions and limitations under the Licence.     *
# *                                                                         *
# ***************************************************************************/

import os

QDCLIB_VERSION = None
QDCLIB_ROOT_DIR = os.path.dirname( os.path.abspath(__file__) )

f = open( os.path.join( QDCLIB_ROOT_DIR, "qdclib_version.txt" ) )

QDCLIB_VERSION = f.read().strip()

f.close()

# # status of 'B'ranch in 'S'hort format
# branch=`git branch --remote --verbose --no-abbrev --contains | sed -rne 's/^[^\/]*\/([^\ ]+).*$/\1/p'`
# branch="${branch#HEAD}"
# if [[ -z ${branch} || "${branch}" =~ ^HEAD ]]; then
#   # unpushed stuff, lets try rev-parse instead
#  branch=`git rev-parse --abbrev-ref HEAD`

# git rev-list --full-history --all --abbrev-commit | wc -l
# git rev-list --count HEAD 
# https://stackoverflow.com/questions/4120001/what-is-the-git-equivalent-for-revision-number
# 
#

def get_git_revision():
    """
    empty doc

    Returns
    -------
    None.

    """
    return None

def get_version_info():
    """
    empty doc

    Returns
    -------
    QDCLIB_VERSION : TYPE
        DESCRIPTION.

    """
    return QDCLIB_VERSION

__version__ = get_version_info()