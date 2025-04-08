def get_table_row (idx,figure, dry_paths, wet_paths, x=None, y=None):
    """
    Generate a table row for the given pedal name, gain, tone, and audio paths.
    """    

    return f"""
<tr>
    <td class="align-middle">{idx}</td>
    <td class="align-middle">
    {"" if x is None else f"x={x:.1f} y={y:.1f}<br>"}
    <img src="{figure}" width="300 rem">
    </td>
    <td class="align-middle">
      <b>Amp: </b><a href="{dry_paths[1]}">{os.path.basename(dry_paths[1])}</a><br>
      <audio controls>
          <source src="{dry_paths[1]}"
          type="audio/mp3">
      </audio><br>
      <b>DI: </b><a href="{dry_paths[0]}">{os.path.basename(dry_paths[0])}</a><br>
      <audio controls>
          <source src="{dry_paths[0]}"
          type="audio/mp3">
      </audio>
    </td>
    <td class="align-middle">
      <b>Amp: </b><a href="{wet_paths[1]}">{os.path.basename(wet_paths[1])}</a><br>
      <audio controls>
          <source src="{wet_paths[1]}"
          type="audio/mp3">
      </audio><br>
      <b>DI: </b><a href="{wet_paths[0]}">{os.path.basename(wet_paths[0])}</a><br>
      <audio controls>
          <source src="{wet_paths[0]}"
          type="audio/mp3">
      </audio>
    </td>
</tr>"""


HEADER = """
<!DOCTYPE html>
<html lang="en">

  <head>
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta name="description" content="Hybrid neural guitar overdrive">
    <meta name="keywords"
      content="neural audio effects, deep learning, neural networks">
    <meta name="author" content="Domenico Stefani">

    <title>Morphdrive Hybrid Demos</title>
    <!-- Bootstrap core CSS -->
    <!--link href="bootstrap.min.css" rel="stylesheet"-->
    <link rel="stylesheet"
      href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css"
      integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm"
      crossorigin="anonymous">

    <style>
        .table-borderless td, .table-borderless th {
            border: none;
        }
        .progtab {
            margin: 0 auto;
            padding: 0;
            width: 100%;
        }
        .progtab tr {
            margin: 0;
            padding: 0;
        }
        .progtab td {
            margin: 0;
            padding: 0;
        }

        .left{
            width: 30%;
        }
    </style>
  </head>

  <body>
    <div class="text-center">
      <div class="container"></div>
      <h1 class="pt-5">Morphdrive: Demos for hybrid configurations</h1>
      <p style="font-size: 1.3em; margin-bottom: 0;">
        <a href="https://github.com/return-nihil"> Ardan Dal Rì</a>,
        <a href="http://www.domenicostefani.com"> Domenico Stefani</a>,
        Luca Turchet,
        Nicola Conci
      </p>
      <p style="margin-top: 0;">
        University of Trento, Italy
      </p>

      <h5 class style="color: black;" role="group" aria-label="Top menu">
        <a href="index.html">>> Back to the Homepage <<</a>
      </h5>
      <a href="demos-real.html">> to existing configuration demos <<</a>

      <!-- <div class="mt-3">
        <img class="img-fluid" src='images/morphdrive-arch.svg' width="20%">
        <img class="img-fluid" src='images/morphdrive_gui.png' width="20%">
        <img class="img-fluid" src="images/robot-gif.gif" width="20%">
      </div> -->
    </div>

    <div class="container mt-4 "
      style="max-width: 1250px; margin-left: auto; margin-right: auto;">
      <div class="container">
      

        <div id="examples" class="section mt-4">
          <div class="row text-center justify-content-center">
            <div class="table-responsive">

            
              <table class="align-middle text-center table">
                <tr>
                  <th>#</th>
                  <th>2D Conditioning over THD map</th>
                  <th>Dry</th>
                  <th>Static HyperRNN + VAE</th>
                </tr>
""".encode('utf-8')


FOOTER = """
</table>

            </div>
          </div>
        </div>

        

        <hr>
        <footer>
          <p><!--*Accepted to the <a href="https://neuripscreativityworkshop.github.io/2021/">NeurIPS 2021 Workshop on Machine
            Learning for Creativity and Design</a> -->
            <br> Send feedback and questions to <a href="https://github.com/return-nihil">Ardan Dal Rì</a> and <a href="http://domenicostefani.com">Domenico Stefani</a>.
            <br>
            <i>Website inspired by <a href="https://csteinmetz1.github.io/steerable-nafx/">Steerable Neural Audio Effects</a> by Christian J. Steinmetz</i>
          </p>


        </footer>
      </div>
    </div>

    <script type="text/javascript"
      src="https://cdn.rawgit.com/pcooksey/bibtex-js/5ccf967/src/bibtex_js.js"></script>
    <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"
      integrity="sha384-DfXdz2htPH0lsSSs5nCTpuj/zy4C+OGpamoFVy38MVBnE+IbbVYUew+OrCXaRkfj"
      crossorigin="anonymous"></script>
    <script
      src="https://cdn.jsdelivr.net/npm/popper.js@1.16.0/dist/umd/popper.min.js"
      integrity="sha384-Q6E9RHvbIyZFJoft+2mJbHaEWldlvI9IOYy5n3zV9zzTtmI3UksdQRVvoxMfooAo"
      crossorigin="anonymous"></script>
    <script
      src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/js/bootstrap.min.js"
      integrity="sha384-OgVRvuATP1z7JjHLkuOU7Xw704+h835Lr+6QL9UvYjZE3Ipu6Tp75j7Bh/kR0JKI"
      crossorigin="anonymous"></script>

  </body>

</html>
""".encode('utf-8')


import os
os.chdir(os.path.abspath(os.path.dirname(__file__)))
from glob import glob
from utils import DBmetadataRenamer

dry_files = sorted(glob(r"demo\dry\*.mp3"))
morphdrive_files = sorted(glob(r"demo\morphdrive_hybrid\*.mp3"), reverse=True)
# print('morphdrive_files', morphdrive_files)

# print(dbwet_files[0])

output_lines = []
output_lines.append(HEADER)


datasetMetadataRenamer = DBmetadataRenamer()

mainpage_table_lines = []
INTERESTING_CONFIGS = [
    'dcb_wet_guitar-funk_x0.7_y0.8.mp3',
    'dcb_wet_guitar-chords_x0.2_y0.1.mp3',
    'dcb_wet_epiano_x0.7_y0.8.mp3',
    'dcb_wet_guitar-chords_x0.9_y0.1.mp3',
    'dcb_wet_organ_x0.7_y0.8.mp3',
]

prevname = ""
for iw, wet_path in enumerate(morphdrive_files):
    # print('wet_path',wet_path)
    dbfile = os.path.join('demo','dataset',os.path.basename(wet_path).removeprefix('wet_'))
    # print(dbfile)

    dryfile = None
    for dry_path in dry_files:
        if os.path.splitext(os.path.basename(dry_path))[0]+'_' in dbfile:
            dryfile = dry_path
            break
    assert dryfile is not None, f"File {dbfile} not found in dry folder"
    # print(dryfile)

    # print('wet_path',wet_path)
    # print(os.path.basename(wet_path).removeprefix('dcb_wet_').removesuffix('.mp3').split('_')[1])
    xval = float(os.path.basename(wet_path).removeprefix('dcb_wet_').removesuffix('.mp3').split('_')[1].removeprefix('x'))
    yval = float(os.path.basename(wet_path).removeprefix('dcb_wet_').removesuffix('.mp3').split('_')[2].removeprefix('y'))

    mappath = os.path.join(os.path.dirname(wet_path), f'heatmap_point_{xval:.1f}_{yval:.1f}.png')
    assert os.path.exists(mappath), f"File {mappath} not found"

    # if pedal_name != prevname:
    #     prevname = pedal_name
    #     if iw > 0:
    #         output_lines.append('<tr><td class="v-divider"></td><td class="v-divider"></td><td class="v-divider"></td><td class="v-divider"></td></tr>'.encode('utf-8'))


    # Filenames with amp sym 
    dryfiles = (dryfile,os.path.join(os.path.dirname(dryfile),'amp','amp_'+os.path.basename(dryfile)))
    assert os.path.exists(dryfiles[1]), f"File {dryfiles[1]} not found in amp folder"

    wet_paths = (wet_path,os.path.join(os.path.dirname(wet_path),'amp','amp_'+os.path.basename(wet_path).removeprefix('dcb_')))
    assert os.path.exists(wet_paths[1]), f"File {wet_paths[1]} not found in amp folder"

    row = get_table_row(iw+1,mappath,dryfiles,wet_paths, x=xval, y=yval).encode('utf-8')
    output_lines.append(row)
    if os.path.basename(wet_path) in INTERESTING_CONFIGS:
        mainpage_table_lines.append(row)
    
output_lines.append(FOOTER)

with open('demos-hybrid.html', 'wb') as f:
    f.writelines(output_lines)

with open('demos-hybrid-mainpage.html', 'wb') as f:
    f.writelines(mainpage_table_lines)
    


    