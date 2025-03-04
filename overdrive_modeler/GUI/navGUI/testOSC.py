from pythonosc import udp_client
import keyboard
import random, os

labelcounter = 0
points_per_label = 5

import pandas as pd

DATAPATH = "../../network/complete_dataframe_8.csv"
assert os.path.exists(DATAPATH), "File not found at %s" % DATAPATH
df = pd.read_csv(DATAPATH)

# // Create a new dataframe extracting only the columns we need (coords, label)
df2 = df[['coords', 'label']]
# Coords is a string of the form "  (0.9365225, 0.18346773)" so r"[ ]*\(([^,]+),([^)]+)\)[ ]*" extracts the two numbers
# df2['x'] = df2['coords'].str.extract(r"[ ]*\(([^,]+),[^)]+\)[ ]*").astype(float)
# df2['y'] = df2['coords'].str.extract(r"[ ]*\([^,]+,([^)]+)\)[ ]*").astype(float)
# redo using dfx.loc[row_indexer, col_indexer]
df2['x'] = df2['coords'].apply(lambda x: float(x.split(',')[0].strip()[1:]))
df2['y'] = df2['coords'].apply(lambda x: float(x.split(',')[1].strip()[:-1]))
# Drop the coords column
df2 = df2.drop(columns=['coords'])
# Sort by label
df2 = df2.sort_values(by='label')
# add a labelidx column created from unique labels
df2['labelidx'] = df2['label'].astype('category').cat.codes

print('Read %d pedal data points from %s' % (len(df2),os.path.basename(DATAPATH)))
  
# print(df2)

def send_osc_message():
    global labelcounter
    # Initialize OSC client
    client = udp_client.SimpleUDPClient("127.0.0.1", 12000)
    
    print("Press:\n'a' to clear point array\n's' to render bacground\n'd' to send all dataset\nPress Ctrl+C to exit.")
    
    try:
        # Set up event handler for 'a' key
        while True:
            # if keyboard.is_pressed('a'):
            #     client.send_message("/test", [42, 3.14, "Hello"])
            #     print("Message sent: 42, 3.14, Hello")
            #     # Wait for key release to avoid multiple sends
            #     keyboard.wait('a', suppress=True, trigger_on_release=True)

            
            if keyboard.is_pressed('a'):
                client.send_message("/clearPoints", [])
                print("Message sent: /clearPoints")
                keyboard.wait('a', suppress=True, trigger_on_release=True)

            
            if keyboard.is_pressed('s'):
                client.send_message("/renderBackground", [])
                print("Message sent: /renderBackground")
                keyboard.wait('s', suppress=True, trigger_on_release=True)
            
            
            if keyboard.is_pressed('d'):
                for index, row in df2.iterrows():
                    x = row['x']
                    y = row['y']
                    labelidx = row['labelidx']
                    labelname = row['label']
                    labelcounter += 1
                    client.send_message("/addPoint", [x,y,labelidx,labelname])
                    print("Message sent: /addPoint %f %f %d '%s'" % (x, y, labelidx,labelname))
                # randx = random.uniform(0, 1)
                # randy = random.uniform(0, 1)
                # randlabel = labelcounter % points_per_label
                # name = "ciao"+str(labelcounter)
                # labelcounter += 1
                # client.send_message("/addPoint", [randx,randy,randlabel,name])
                # print("Message sent: /addPoint %f %f %d '%s'" % (randx, randy, randlabel,name))
                keyboard.wait('d', suppress=True, trigger_on_release=True)

            
            # if keyboard.is_pressed('d'):
            #         randx = random.uniform(0, 1)
            #         randy = random.uniform(0, 1)
            #         randlabel = labelcounter % points_per_label
            #         name = "ciao"+str(labelcounter)
            #         labelcounter += 1
            #         client.send_message("/addPoint", [randx,randy,randlabel,name])
            #     print("Message sent: /addPoint %f %f %d '%s'" % (randx, randy, randlabel,name))
            #     keyboard.wait('d', suppress=True, trigger_on_release=True)
            
    except KeyboardInterrupt:
        print("\nStopped sending messages.")

if __name__ == "__main__":
    localcounter = 0
    send_osc_message()