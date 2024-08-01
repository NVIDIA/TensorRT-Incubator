
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import re
import subprocess
import sys
from datetime import datetime

current_year = str(datetime.now().year)

def get_license_header():
    for license_path in ['LICENSE', 'tripy/LICENSE']:
        if os.path.exists(license_path):
            with open(license_path, 'r', encoding='utf-8') as license_file:
                return license_file.read().strip()
    raise FileNotFoundError('LICENSE file not found in the current directory or tripy folder')

license_text = get_license_header()

def update_file(file_path):
    with open(file_path, 'r+', encoding='utf-8') as file:
        content = file.read()
        copyright_pattern = r"Copyright \(c\) 1993-\d{4}"
        if re.search(copyright_pattern, content):
            updated_content = re.sub(copyright_pattern, f'Copyright (c) 1993-{current_year}', content)
        else:
            updated_content = license_text + '\n' + content

        if content != updated_content:
            file.seek(0)
            file.write(updated_content)
            file.truncate()
            return True
    return False

def get_files(mode):
    if mode == 'new':
        command = ['git', 'diff', '--cached', '--name-only', '--diff-filter=A']
    elif mode == 'all':
        command = ['git', 'ls-files']
    else:
        raise ValueError("Invalid mode. Use 'new' or 'all'.")
    
    result = subprocess.run(command, capture_output=True, text=True)
    return [f for f in result.stdout.splitlines() if f.endswith('.py') and f.startswith('tripy/')]

def main(mode):
    files = get_files(mode)
    updated_files = [f for f in files if update_file(f)]
    
    if mode == 'new' and updated_files:
        subprocess.run(['git', 'add'] + updated_files)
    
    print(f"Updated {len(updated_files)} out of {len(files)} Python files in '{mode}' mode.")

if __name__ == '__main__':
    import sys
    mode = 'all' if len(sys.argv) > 1 and sys.argv[1] == 'all' else 'new'
    main(mode)
