@description('Specifies the name of the deployment.')
param name string

@description('Specifies the name of the environment.')
param environment string

@description('Specifies the location of the Azure Machine Learning workspace and dependent resources.')
param location string = resourceGroup().location

@description('The minimum number of nodes to use on the cluster. If not specified, defaults to 0')
param minNodeCount int = 0
@description(' The maximum number of nodes to use on the cluster. If not specified, defaults to 4.')
param maxNodeCount int = 1

@description('Specifies whether to reduce telemetry collection and enable additional encryption.')
param hbi_workspace bool = false

@description(' The size of agent VMs. More details can be found here: https://aka.ms/azureml-vm-details.')
param vmSize string = 'Standard_NC6s_v3'

var tenantId = subscription().tenantId
var storageAccountName_var = 'st${name}${environment}'
var keyVaultName_var = 'kv-${name}-${environment}'
var applicationInsightsName_var = 'appi-${name}-${environment}'
var containerRegistryName_var = 'cr${name}${environment}'
var workspaceName_var = 'mlw${name}${environment}'
var clusterName_var = 'clust${name}${environment}'
var storageAccount = storageAccountName.id
var keyVault = keyVaultName.id
var applicationInsights = applicationInsightsName.id
var containerRegistry = containerRegistryName.id

resource storageAccountName 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: storageAccountName_var
  location: location
  sku: {
    name: 'Standard_RAGRS'
  }
  kind: 'StorageV2'
}

resource keyVaultName 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: keyVaultName_var
  location: location
  properties: {
    sku: {
      family: 'A'
      name: 'standard'
    }
    tenantId: tenantId
    accessPolicies: []
    enableSoftDelete: true
  }
}

resource applicationInsightsName 'Microsoft.Insights/components@2020-02-02' = {
  name: applicationInsightsName_var
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
  }
}

resource containerRegistryName 'Microsoft.ContainerRegistry/registries@2023-11-01-preview' = {
  name: containerRegistryName_var
  location: location
  sku: {
    name: 'Standard'
  }
  properties: {
    adminUserEnabled: true
  }
}

resource workspaceName 'Microsoft.MachineLearningServices/workspaces@2023-10-01' = {
  identity: {
    type: 'SystemAssigned'
  }
  name: workspaceName_var
  location: location
  properties: {
    friendlyName: workspaceName_var

    storageAccount: storageAccount
    keyVault: keyVault
    applicationInsights: applicationInsights
    containerRegistry: containerRegistry
    hbiWorkspace: hbi_workspace
  }
}


resource clusterName 'Microsoft.MachineLearningServices/workspaces/computes@2021-01-01' = {
  parent: workspaceName
  name: clusterName_var
  location: location
  properties: {
    computeType: 'AmlCompute'
    properties: {
      vmSize: vmSize
      scaleSettings: {
        minNodeCount: minNodeCount
        maxNodeCount: maxNodeCount
      }
    }
  }
}
