import os
import re


def include_packages(root_packages_list):
    """
    This method generates a list of all package names to include
    starting from a list of root package names.
    :param root_packages_list: List of root package names to include
    :return: Returns the list of all package names to include
    """

    packages_to_include = list()

    for root_package in root_packages_list:
        for root, dirs, files in os.walk(root_package):
            if '__init__.py' in files:
                packages_to_include.append(re.sub('^[^A-z0-9_]+', '', root.replace('/', '.')))

    return packages_to_include

pks = include_packages(['tf_3dmm', 'sample'])
print(pks)