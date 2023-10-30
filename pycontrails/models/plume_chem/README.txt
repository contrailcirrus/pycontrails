Plume chem pull request 27/10/23

Updates:
- Working on a script init_chem.py, which introduces the ChemDataset class. This class is to be used for the background chemistry dataset. 
-The purpose of this script is to initialise as much chemistry data as possible, pre time integration. E.g. local times, solar zenith angles, bg species concs, photolysis params etc.
- So far, I have managed to initialise a ChemDataset, subject to time and spatial inputs (bounds and resolutions). This is currently setting up the xarray dataset with zeros, and then populating local_time and solar zenith angles, for all boxes and all timesteps.
- Currently trying to write a _get_species method that gets bg species data for all time and positions, however having a hard time thinking how best to import the data.

Instructions:
Run boxm_test.ipynb (sorry its still a notebook...) and this will make the necessary calls to initialise and start populating the chem dataset, pre integration.

Todo:
[] Write _get_species method so that it takes input of local times and grid cells, and outputs the same xarray dataset with species concs in "Y" populated. Need help thinking of best waty to do it, as currently at three for loops. Yuck!
[] Write tests to test output of each of the methods being called in public open_chemdataset method. E.g. _init_chem() should definitely print all zeros. Can also check that species concs inputs are the same every time.
[X] Convert test script from notebook to python script.
