import pandas as pd
import os

DATASET_METADATA_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','overdrive_modeler','network','dataset','dataset_metadata.csv'))
assert os.path.exists(DATASET_METADATA_FILE), f"Dataset metadata file not found at {DATASET_METADATA_FILE}"

def get_dataset_metadata():
    return pd.read_csv(DATASET_METADATA_FILE)

class DBmetadataRenamer:
    def __init__(self):
        self.metadata = get_dataset_metadata()

    def datasetname2shortname(self,datasetname):
        if datasetname in self.metadata['datasetname'].values:
            return self.metadata[self.metadata['datasetname']==datasetname]['shortname'].values[0]
        else:
            print(f"Dataset name {datasetname} not found in metadata")
            return None
        
    def datasetname2model(self,datasetname):
        if datasetname in self.metadata['datasetname'].values:
            return self.metadata[self.metadata['datasetname']==datasetname]['model'].values[0]
        else:
            print(f"Dataset name {datasetname} not found in metadata")
            return None
        
    def shortname2datasetname(self,shortname):
        if shortname in self.metadata['shortname'].values:
            return self.metadata[self.metadata['shortname']==shortname]['datasetname'].values[0]
        else:
            print(f"Short name {shortname} not found in metadata")
            return None
        
    def shortname2brandmodel(self,shortname):
        if shortname in self.metadata['shortname'].values:
            return self.metadata[self.metadata['shortname']==shortname][['brand','model']].values[0]
        else:
            print(f"Short name {shortname} not found in metadata")
        
