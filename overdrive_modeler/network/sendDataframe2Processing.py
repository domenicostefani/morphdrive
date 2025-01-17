from pythonosc import udp_client
import time
import os
import pandas as pd

# Load the data from C:\Users\cimil\Desktop\DAFx24-Pedaliny\ROBOT_RECORDER\overdrive_modeler\network\complete_dataframe_8.csv
TEST_DATAPATH = "../../network/complete_dataframe_8.csv"

class DFSender:
    def __init__(self, ip_toSendTo, port_toSendTo):
        self.df2 = None
        self.client = udp_client.SimpleUDPClient(ip_toSendTo, port_toSendTo)
        
    def readDataframe(self,DATAPATH):
        assert os.path.exists(DATAPATH), "File not found at %s" % DATAPATH
        df = pd.read_csv(DATAPATH)

        # // Create a new dataframe extracting only the columns we need (coords, label)
        self.df2 = df[['coords', 'label', 'name']]
        self.df2['x'] = self.df2['coords'].apply(lambda x: float(x.split(',')[0].strip()[1:]))
        self.df2['y'] = self.df2['coords'].apply(lambda x: float(x.split(',')[1].strip()[:-1]))
        # Drop the coords column
        self.df2 = self.df2.drop(columns=['coords'])

        self.df2['gain'] = self.df2['name'].apply(lambda x: float(x.split('_')[0][1])/5.0)
        self.df2['tone'] = self.df2['name'].apply(lambda x: float(x.split('_')[1][1])/5.0)
        # Drop the name column
        self.df2 = self.df2.drop(columns=['name'])

        # Sort by label
        self.df2 = self.df2.sort_values(by='label')
        # add a labelidx column created from unique labels
        self.df2['labelidx'] = self.df2['label'].astype('category').cat.codes

        print('Read %d pedal data points from %s' % (len(self.df2),os.path.basename(DATAPATH)))
  
    def send(self):
        self.client.send_message("/clearPoints", [])
        print("Message sent: /clearPoints")
        time.sleep(0.01)
                    
        for _ , row in self.df2.iterrows():
            x = row['x']
            y = row['y']
            labelidx = row['labelidx']
            labelname = row['label'].replace('Nembrini','')
            gain = row['gain']
            tone = row['tone']
            self.client.send_message("/addPoint", [x,y,labelidx,labelname,gain,tone])
            print("Message sent: /addPoint %f %f %d '%s' %f %f" % (x, y, labelidx,labelname,gain,tone))
            time.sleep(0.01)

        self.client.send_message("/renderBackground", [])
        print("Message sent: /renderBackground")


if __name__ == "__main__":
    sender = DFSender("127.0.0.1", 12000)
    sender.readDataframe(TEST_DATAPATH)
    sender.send()