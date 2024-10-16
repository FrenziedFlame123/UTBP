
# IMPORTS

import pathlib
import os
import string
import glob
import pandas as pd
import numpy as np
import papermill as pm
import pandas_flavor as pf
from tqdm import tqdm
from traitlets.config import Config
from nbconvert.preprocessors import TagRemovePreprocessor
from nbconvert.exporters import HTMLExporter, PDFExporter
from nbconvert.writers import FilesWriter

#---------------------------------------------------------------

# run_reports()
# required packages: 
#   import pathlib
#   import os
#   import string
#   import glob
#   import pandas as pd
#   import numpy as np
#   import papermill as pm
#   import pandas_flavor as pf
#   from tqdm import tqdm
#   from traitlets.config import Config
#   from nbconvert.preprocessors import TagRemovePreprocessor
#   from nbconvert.exporters import HTMLExporter, PDFExporter
#   from nbconvert.writers import FilesWriter

@pf.register_dataframe_method
def run_reports(
    df,
    id_sets = None,
    report_titles = None,
    template = "",
    directory = "",
    convert_to_html = False,
    convert_to_pdf = False
):
    """
    Generate and execute Papermill Jupyter Notebooks based on a provided template, 
    with options to convert the generated notebooks into HTML and/or PDF formats.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data to be passed to the notebooks.
    id_sets : list of list of str or int, optional
        A list of lists where each sublist contains identifiers to be used in the 
        report generation. Each set of identifiers will be passed as parameters 
        to the notebook.
    report_titles : list of str, optional
        A list of titles for the generated reports. The titles will also be used 
        to generate the filenames for the notebooks.
    template : str, optional
        The file path to the Papermill Jupyter Notebook template. This notebook 
        will be used as a base for generating the reports.
    directory : str, optional
        The directory where the generated notebooks and converted files will be 
        saved. If the directory does not exist, it will be created.
    convert_to_html : bool, optional
        If True, the generated notebooks will be converted to HTML format.
    convert_to_pdf : bool, optional
        If True, the generated notebooks will be converted to PDF format.

    Notes
    -----
    - If the specified template does not exist, the function will print a message 
      indicating the missing template.
    - If the specified directory does not exist, it will be created.
    - The generated notebooks will be saved with filenames derived from the 
      `report_titles`, with punctuation removed and spaces replaced by underscores.
    - The function uses the Papermill library to execute the notebooks and the 
      nbconvert library to convert the notebooks into HTML and PDF formats.
    
    Examples
    --------
    
    run_reports(
        df=my_dataframe,
        id_sets=[['id1', 'id2'], ['id3', 'id4']],
        report_titles=['Report 1', 'Report 2'],
        template='path/to/template.ipynb',
        directory='path/to/save/reports',
        convert_to_html=True,
        convert_to_pdf=False
    )
    
    ---
    
    df.run_reports(
        id_sets = id_sets,
        report_titles = titles,
        template = "jupyter_papermill/template/jupyter_report_template.ipynb",
        directory = "jupyter_html_pdf/sales_reports/",
        convert_to_html = True,
        convert_to_pdf = True
    )
    
    """

    # Make directory and template if not created
    
    temp_path = pathlib.Path(template)
    
    dir_path = pathlib.Path(directory)
    directory_exists = os.path.isdir(dir_path)
    
    if not temp_path:
        print(f"Template you provided does not exist at {temp_path}.")
    
    if not directory_exists:
        print(f"Making directory at {str(dir_path.absolute())}")
        os.makedirs(dir_path) 
    
    # Make Papermill Jupyter Notebooks
    print("Executing Papermill Jupyter Reports...")
    for i, id_set in enumerate(id_sets):

        # Output Path
        report_title = report_titles[i]
        
        file_name = report_title \
            .translate(
            str.maketrans("", "", string.punctuation)
            ) \
            .lower() \
            .replace(" ", "_")

        
        output_path = pathlib.Path(f"{directory}/{file_name}.ipynb")
        
        # Parameters
        params = {
            "ids": id_set,
            "title": report_title,
            "df": df.to_json()
        }
        
        # Papermill Execute
        pm.execute_notebook(
            input_path = temp_path,
            output_path = output_path,
            parameters = params,
            report_mode = True
        )

    #  >>>> NBCONVERT <<<< ----

    # Prep for Conversions
    files = glob.glob(f"{directory}/*.ipynb")
    
    c = Config()
    c.TemplateExporter.exclude_input = True
    c.TagRemovePreprocessor.remove_cell_tags = ("remove_cell",)
    c.TagRemovePreprocessor.remove_all_outputs_tags = ("remove_output",)
    c.TagRemovePreprocessor.remove_input_tags = ("remove_input",)
    c.TagRemovePreprocessor.enabled = True
    
    c.HTMLExporter.preprocessors = ["nbconvert.preprocessors.TagRemovePreprocessor"]
    c.FilesWriter.build_directory = str(directory)
    fw = FilesWriter(config = c)

    # Convert to HTML
    if convert_to_html:
        print("Executing HTML Reports...")
        for file in tqdm(files):
            file_path = pathlib.Path(file)
            file_name = file_path.stem
            file_dir = file_path.parents[0]

            (body, resources) = HTMLExporter(config = c).from_filename(file_path)
            file_dir_html = str(file_dir) + "_html"
            c.FilesWriter.build_directory = str(file_dir_html)
            fw = FilesWriter(config = c)
            fw.write(body, resources, notebook_name = file_name)

    # Convert to PDF
    if convert_to_pdf:
        print("Executing PDF Reports...")
        for file in tqdm(files):
            file_path = pathlib.Path(file)
            file_name = file_path.stem
            file_dir = file_path.parents[0]
            
            (body, resources) = PDFExporter(config = c).from_filename(file_path)
            file_dir_pdf = str(file_dir) + "_pdf"
            c.FilesWriter.build_directory = str(file_dir_pdf)
            fw = FilesWriter(config = c)
            fw.write(body, resources, notebook_name = file_name)

    print("Reporting Complete.")

################################# END OF FUNCTION #################################