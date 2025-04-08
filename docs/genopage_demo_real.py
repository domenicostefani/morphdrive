def get_table_row (pedal_name, gain, tone, dry_path, dataset_path, wet_path):
    """
    Generate a table row for the given pedal name, gain, tone, and audio paths.
    """
    assert gain >= 0 and gain <= 5, "Gain must be between 0 and 5"
    assert tone >= 0 and tone <= 5, "Tone must be between 0 and 5"
    return f"""
<tr>
    <td class="align-middle">
    {pedal_name}
    <table class="progtab text-center table table-borderless align-middle">
        <tr><td class='align-middle left'>Gain</td> <td class='align-middle'><div class="progress"><div class="progress-bar bg-danger" style="width: {gain/5*100}%;">{gain}/5</div></div></td></tr>
        <tr><td class='align-middle left'>Tone</td> <td class='align-middle'><div class="progress"><div class="progress-bar bg-warning" style="width: {tone/5*100}%;">{tone}/5</div></div></td></tr>
    </table>
    <!--Gain <div class="progress"><div class="progress-bar" style="width: {gain/5*100}%;">{gain}/5</div></div><br>
    Tone <div class="progress"><div class="progress-bar bg-info" style="width: {tone/5*100}%;">{tone}/5</div></div><br> -->
    </td>
    <td class="align-middle">
    <audio controls>
        <source src="{dry_path}"
        type="audio/mp3">
    </audio>
    </td>
    <td class="align-middle">
    <audio controls>
        <source src="{dataset_path}"
        type="audio/mp3">
    </audio>
    </td>
    <td class="align-middle">
    <audio controls>
        <source src="{wet_path}"
        type="audio/mp3">
    </audio>
    </td>
</tr>
"""


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

    <title>Morphdrive Real Demos</title>
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
      <h1 class="pt-5">Morphdrive: Demos for real configurations</h1>
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
        <a href="index.html">>> Back to the Homepage<<</a>
      </h5>
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

              


              <table class="text-center table">
                <tr>
                  <th>Pedal/Settings</th>
                  <th>Dry</th>
                  <th>Dataset</th>
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
dbwet_files = sorted(glob(r"demo\dataset\*.mp3"))
morphdrive_files = sorted(glob(r"demo\morphdrive_real\*.mp3"))
# print('morphdrive_files', morphdrive_files)

# print(dbwet_files[0])

output_lines = []
output_lines.append(HEADER)

datasetMetadataRenamer = DBmetadataRenamer()

prevname = ""
for iw, wet_path in enumerate(morphdrive_files):
    # print('wet_path',wet_path)
    dbfile = os.path.join('demo','dataset',os.path.basename(wet_path).removeprefix('wet_'))
    # print(dbfile)
    assert dbfile in dbwet_files, f"File {dbfile} not found in dataset folder"

    dryfile = None
    for dry_path in dry_files:
        if os.path.basename(dry_path) in dbfile:
            dryfile = dry_path
            break
    assert dryfile is not None, f"File {dbfile} not found in dry folder"

    print(wet_path)
    pedal_name = os.path.basename(wet_path).removeprefix('wet_a_').removesuffix('.mp3').split('_')[0]
    # pedal_name = datasetMetadataRenamer.datasetname2model(pedal_name)
    brand,pedal_name = datasetMetadataRenamer.shortname2brandmodel(datasetMetadataRenamer.datasetname2shortname(pedal_name))

    print("os.path.basename(wet_path).removeprefix('wet_a_').split('_')[2]",os.path.basename(wet_path).removeprefix('wet_a_').split('_')[1])
    gain = int(os.path.basename(wet_path).removeprefix('wet_a_').split('_')[1].removeprefix('g'))
    tone = int(os.path.basename(wet_path).removeprefix('wet_a_').split('_')[2].removeprefix('t'))

    if pedal_name != prevname:
        prevname = pedal_name
        if iw > 0:
            output_lines.append('<tr><td class="v-divider"></td><td class="v-divider"></td><td class="v-divider"></td><td class="v-divider"></td></tr>'.encode('utf-8'))

    output_lines.append(get_table_row(pedal_name+f' ({brand})', gain, tone, dryfile, dbfile, wet_path).encode('utf-8'))
    
output_lines.append(FOOTER)

with open('demos-real.html', 'wb') as f:
    f.writelines(output_lines)

    


    