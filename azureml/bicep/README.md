### HOW TO DEPLOY TO AZURE USING BICEP SCRIPTS AND COMMAND LINES

Execute the following commands on your terminal(mac)

#### STEP 1: SET THE REQUIRED VARIABLES
current_date=$(date +%m-%d-%Y)

#### STEP 2: SET THE DEPLOYMENT NAME
deployment_name="<name_as_per_specification>""$current_date"

example for reference: 'AutoRAML02-26-2024'

#### STEP 3: TRIGGER DEPLOYMENT
az deployment group create --name $deployment_name --resource-group <resource_group_name on azure> --template-file ./main.bicep --parameters ./azuredeploy.parameters.json --verbose --confirm-with-what-if

#### STEP 4: CONFIRM CHANGES (y/N)

#### STEP 5: TRACK THE STATUS OF YOUR DEPLOYMENT UNDER
    - https://portal.azure.com/#view/HubsExtension/BrowseResourceGroups
    - Click on your resource group
    - Click on the 'Deployments' tab on the left panel
    - Check the status of the deployment name for which your triggered deployment.

Further references:
https://medium.com/codex/using-bicep-to-create-workspace-resources-and-get-started-with-azure-machine-learning-bcc57fd4fd09
https://medium.com/@dmitri.mahayana/creating-virtual-assistance-using-with-llama-2-7b-chat-model-9f693c8250ee
