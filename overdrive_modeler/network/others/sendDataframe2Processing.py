from pythonosc import udp_client
import time
import os
import pandas as pd

# TEST_DATAPATH = "../../network/complete_dataframe_8.csv"
import argparse

parser = argparse.ArgumentParser(description='Send a dataframe to Processing')
parser.add_argument('DATAPATH', type=str, help='Path to the dataframe to send')

args = parser.parse_args()
TEST_DATAPATH = args.DATAPATH
assert os.path.exists(TEST_DATAPATH), "File not found at %s" % TEST_DATAPATH

class DFSender:
    def __init__(self, ip_toSendTo, port_toSendTo):
        self.df2 = None
        self.client = udp_client.SimpleUDPClient(ip_toSendTo, port_toSendTo)
        
    def readDataframe(self,DATAPATH):
        assert os.path.exists(DATAPATH), "File not found at %s" % DATAPATH
        df = pd.read_csv(DATAPATH)

        if 'label' not in df.columns or 'name' not in df.columns:
            assert 'label_name' in df.columns, "label_name column not found in dataframe"
            assert 'gain' in df.columns, "gain column not found in dataframe"
            assert 'tone' in df.columns, "tone column not found in dataframe"

        # // Create a new dataframe extracting only the columns we need (coords, label)
        if 'label' not in df.columns or 'name' not in df.columns:
            self.df2 = df[['coords', 'label_name', 'gain', 'tone']]
        else:
            self.df2 = df[['coords', 'label', 'name']]
        # Extract x and y from coords
        self.df2['x'] = self.df2['coords'].apply(lambda x: float(x.split(',')[0].strip()[1:]))
        self.df2['y'] = self.df2['coords'].apply(lambda x: float(x.split(',')[1].strip()[:-1]))
        # Drop the coords column
        self.df2 = self.df2.drop(columns=['coords'])

        if 'gain' in self.df2.columns:
            self.df2['gain'] = self.df2['gain'].apply(lambda x: float(x))
        else:
            self.df2['gain'] = self.df2['name'].apply(lambda x: float(x.split('_')[0][1])/5.0)
        if 'tone' in self.df2.columns:
            self.df2['tone'] = self.df2['tone'].apply(lambda x: float(x))
        else:
            self.df2['tone'] = self.df2['name'].apply(lambda x: float(x.split('_')[1][1])/5.0)
        # Drop the name column
        if 'name' in self.df2.columns:
            self.df2 = self.df2.drop(columns=['name'])

        # Sort by label
        labelcolumn = 'label' if 'label' in self.df2.columns else 'label_name'
        self.df2 = self.df2.sort_values(by=labelcolumn)
        # add a labelidx column created from unique labels
        self.df2['labelidx'] = self.df2[labelcolumn].astype('category').cat.codes

        print('Read %d pedal data points from %s' % (len(self.df2),os.path.basename(DATAPATH)))
  
    def send(self):
        self.client.send_message("/clearPoints", [])
        print("Message sent: /clearPoints")
        time.sleep(0.01)
                    
        for _ , row in self.df2.iterrows():
            x = row['x']
            y = row['y']
            labelidx = row['labelidx']
            labelcolumn = 'label' if 'label' in self.df2.columns else 'label_name'
            labelname = row[labelcolumn]
            gain = row['gain']/max(self.df2['gain'])
            tone = row['tone']/max(self.df2['tone'])
            self.client.send_message("/addPoint", [x,y,labelidx,labelname,gain,tone])
            print("Message sent: /addPoint %f %f %d '%s' %f %f" % (x, y, labelidx,labelname,gain,tone))
            time.sleep(0.01)

        self.client.send_message("/renderBackground", [])
        print("Message sent: /renderBackground")


if __name__ == "__main__":
    sender = DFSender("127.0.0.1", 12000)
    sender.readDataframe(TEST_DATAPATH)
    sender.send()