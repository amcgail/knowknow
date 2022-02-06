# demos coming from https://guides.dataverse.org/en/latest/api/native-api.html#add-file-api

# NOTE just found PyDataverse: https://pydataverse.readthedocs.io/en/latest/_modules/pyDataverse/utils.html
# easy package for doing this stuff (so I don't have to fuck it up so much...)

from . import env
import requests
from pathlib import Path


from requests import ConnectionError


def confirm(pmt = "",):
    """
    Ask user to enter Y or N (case-insensitive).
    :return: True if the answer is Y.
    :rtype: bool
    """
    answer = ""
    while answer not in ["y", "n"]:
        answer = input(f"{pmt} [Y/N]? ").lower()
    return answer == "y"


def download(
    persistent_id,
    dataverse_server = 'https://dataverse.harvard.edu',
    api_key = None
):

    if persistent_id[:4] != 'doi:':
        persistent_id = 'doi:'+persistent_id

    dv_url = f"{dataverse_server}/api/datasets/:persistentId/?persistentId={persistent_id}"
    if api_key is not None:
        dv_url += f"&key={api_key}"

    try:
        r = requests.get(dv_url)
    except ConnectionError:
        return None

    res = r.json()

    if res['status'] != 'OK':
        #raise Exception("There was an issue retrieving the dataset. Please check your persistent ID, and provide an API key if this is a DRAFT version.")
        return None

    v = res['data']['latestVersion']
    title = [x for x in v['metadataBlocks']['citation']['fields'] if x['typeName'] == 'title'][0]['value']

    print(f"Retrieving latest version of {title}. datasetId = {v['datasetId']}. last updated {v['lastUpdateTime']}.")
    print(f"{len(v['files'])} files. Total file size = {sum(x['dataFile']['filesize'] for x in v['files']) / 1000000:0.1f} Mb")
    
    outFold = env.variable_dir.joinpath(title)

    while outFold.exists():
        print(f"The target folder {outFold} exists. Prompting to overwrite.")
        if confirm("Overwrite?"):
            break

        outFoldName = input(f"Enter a new folder to store dataset {title}")
        outFold = env.variable_dir.joinpath(outFoldName)

    if outFold.exists():
        raise Exception("I'm too scared to overwrite. Aborting")

    print("Downloading zip file...")

    down = f"{dataverse_server}/api/access/dataset/:persistentId/?persistentId={persistent_id}"
    headers = {'X-Dataverse-key':api_key}
    r = requests.get(down, headers=headers, allow_redirects=True)

    tmp_file = env.variable_dir.joinpath('tmp.zip')
    tmp_file.open('wb').write( r.content )

    print(f"Done! Unzipping into {outFold.name}")
    import zipfile
    with zipfile.ZipFile(tmp_file, 'r') as zip_ref:
        zip_ref.extractall(outFold)

    tmp_file.unlink()

    return title


def upload( 
    api_key, 
    dataset_name,
    dataset_description="", 
    dataverse_name = 'kk-citation-counts',
    dataverse_server = 'https://dataverse.harvard.edu'):


    to_upload = Path(env.variable_dir).joinpath( dataset_name )
    if not to_upload.exists():
        raise Exception("Please provide a valid `dataset_name`, one which exists. It should be a folder in the `knowknow` variable directory.")
        
    # no trailing slash

    metadata = {
    "datasetVersion": {
        "metadataBlocks": {
        "citation": {
            "fields": [
            {
                "value": dataset_name,
                "typeClass": "primitive",
                "multiple": False,
                "typeName": "title"
            },
            {
                "value": [
                {
                    "authorName": {
                    "value": "McGail, Alec",
                    "typeClass": "primitive",
                    "multiple": False,
                    "typeName": "authorName"
                    },
                    "authorAffiliation": {
                    "value": "Cornell University",
                    "typeClass": "primitive",
                    "multiple": False,
                    "typeName": "authorAffiliation"
                    }
                }
                ],
                "typeClass": "compound",
                "multiple": True,
                "typeName": "author"
            },
            {
                "value": [ 
                    { "datasetContactEmail" : {
                        "typeClass": "primitive",
                        "multiple": False,
                        "typeName": "datasetContactEmail",
                        "value" : "am2873@cornell.edu"
                    },
                    "datasetContactName" : {
                        "typeClass": "primitive",
                        "multiple": False,
                        "typeName": "datasetContactName",
                        "value": "McGail, Alec"
                    }
                }],
                "typeClass": "compound",
                "multiple": True,
                "typeName": "datasetContact"
            },
            {
                "value": [ {
                "dsDescriptionValue":{
                    "value":   f"""\
Count dataset for Web of Science citation data.
{dataset_description}
Produced and uploaded by the Python package <a href='https://pypi.org/project/knowknow-amcgail/'>knowknow</a>""",
                    "multiple":False,
                "typeClass": "primitive",
                "typeName": "dsDescriptionValue"
                }}],
                "typeClass": "compound",
                "multiple": True,
                "typeName": "dsDescription"
            },
            ],
            "displayName": "Citation Metadata"
        }
        }
    }
    }

    import json
    import requests

    # -------------------
    # Start by creating a new dataset
    # -------------------

    url_dataset_id = '%s/api/dataverses/%s/datasets?key=%s' % (dataverse_server, dataverse_name, api_key)

    params_as_json_string = json.dumps(metadata)
    payload = dict(jsonData=params_as_json_string)

    r = requests.post(url_dataset_id, data=params_as_json_string)

    persistentId = r.json()['data']['persistentId']
    dataset_id = r.json()['data']['id']

    print(f"Created dataset with persistent ID = {persistentId}")

    for fn in to_upload.glob("*"):

        # --------------------------------------------------
        # Prepare "file"
        # --------------------------------------------------

        #F_SOURCE = 'D:/knowknow/data/sociology-wos-74b/'
        if 'doc ___ c' in fn.name:
            continue

        print("uploading", fn.name, '...')

        assert fn.exists()

        file_content = fn.open('rb').read()
        files = {'file': (fn.name, file_content)}

        # --------------------------------------------------
        # Using a "jsonData" parameter, add optional description + file tags
        # --------------------------------------------------
        params = dict()
        params_as_json_string = json.dumps(params)
        payload = dict(jsonData=params_as_json_string)

        # --------------------------------------------------
        # Add file using the Dataset's id
        # --------------------------------------------------
        url_dataset_id = '%s/api/datasets/%s/add?key=%s' % (dataverse_server, dataset_id, api_key)

        # -------------------
        # Make the request
        # -------------------
        r = requests.post(url_dataset_id, data=payload, files=files)
