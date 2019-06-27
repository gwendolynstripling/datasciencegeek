
# coding: utf-8

# In[27]:


get_ipython().run_line_magic('matplotlib', 'inline')
import azureml.core
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as scikitlearn
import azureml.dataprep as dprep

import azureml.core
from azureml.core import Workspace

# check core SDK version number
print("Azure ML SDK Version: ", azureml.core.VERSION)


# In[4]:


import azureml.core

print("This notebook was created using version 1.0.23 of the Azure ML SDK")
print("You are currently using version", azureml.core.VERSION, "of the Azure ML SDK")


# ## Configure your Azure ML workspace
# 
# ### Workspace parameters
# 
# To use an AML Workspace, you will need to import the Azure ML SDK and supply the following information:
# * Your subscription id
# * A resource group name
# * (optional) The region that will host your workspace
# * A name for your workspace
# 
# You can get your subscription ID from the [Azure portal](https://portal.azure.com).
# 
# You will also need access to a [_resource group_](https://docs.microsoft.com/en-us/azure/azure-resource-manager/resource-group-overview#resource-groups), which organizes Azure resources and provides a default region for the resources in a group.  You can see what resource groups to which you have access, or create a new one in the [Azure portal](https://portal.azure.com).  If you don't have a resource group, the create workspace command will create one for you using the name you provide.
# 
# The region to host your workspace will be used if you are creating a new workspace.  You do not need to specify this if you are using an existing workspace. You can find the list of supported regions [here](https://azure.microsoft.com/en-us/global-infrastructure/services/?products=machine-learning-service).  You should pick a region that is close to your location or that contains your data.
# 
# The name for your workspace is unique within the subscription and should be descriptive enough to discern among other AML Workspaces.  The subscription may be used only by you, or it may be used by your department or your entire enterprise, so choose a name that makes sense for your situation.
# 
# The following cell allows you to specify your workspace parameters.  This cell uses the python method `os.getenv` to read values from environment variables which is useful for automation.  If no environment variable exists, the parameters will be set to the specified default values.  
# 
# If you ran the Azure Machine Learning [quickstart](https://docs.microsoft.com/en-us/azure/machine-learning/service/quickstart-get-started) in Azure Notebooks, you already have a configured workspace!  You can go to your Azure Machine Learning Getting Started library, view *config.json* file, and copy-paste the values for subscription ID, resource group and workspace name below.
# 
# Replace the default values in the cell below with your workspace parameters

# In[17]:


import os

subscription_id = os.getenv("SUBSCRIPTION_ID", default="f241287f-a1e7-4051-9f53-1e6d74c31cc6")
resource_group = os.getenv("RESOURCE_GROUP", default="science-vm")
workspace_name = os.getenv("WORKSPACE_NAME", default="energy_analytics")
#workspace_region = os.getenv("WORKSPACE_REGION", default="")


# ### Create a new workspace
# 
# If you don't have an existing workspace and are the owner of the subscription or resource group, you can create a new workspace.  If you don't have a resource group, the create workspace command will create one for you using the name you provide.
# 
# **Note**: As with other Azure services, there are limits on certain resources (for example AmlCompute quota) associated with the Azure ML service. Please read [this article](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-manage-quotas) on the default limits and how to request more quota.
# 
# This cell will create an Azure ML workspace for you in a subscription provided you have the correct permissions.
# 
# This will fail if:
# * You do not have permission to create a workspace in the resource group
# * You do not have permission to create a resource group if it's non-existing.
# * You are not a subscription owner or contributor and no Azure ML workspaces have ever been created in this subscription
# 
# If workspace creation fails, please work with your IT admin to provide you with the appropriate permissions or to provision the required resources.

# In[19]:


from azureml.core import Workspace

# Create the workspace using the specified parameters
ws = Workspace.create(name = workspace_name,
                      subscription_id = subscription_id,
                      resource_group = resource_group, 
                      location = workspace_region,
                      create_resource_group = True,
                      exist_ok = True)
ws.get_details()

# write the details of the workspace to a configuration file to the notebook library
ws.write_config()


# ### Create compute resources for your training experiments
# 
# Many of the sample notebooks use Azure ML managed compute (AmlCompute) to train models using a dynamically scalable pool of compute. In this section you will create default compute clusters for use by the other notebooks and any other operations you choose.
# 
# To create a cluster, you need to specify a compute configuration that specifies the type of machine to be used and the scalability behaviors.  Then you choose a name for the cluster that is unique within the workspace that can be used to address the cluster later.
# 
# The cluster parameters are:
# * vm_size - this describes the virtual machine type and size used in the cluster.  All machines in the cluster are the same type.  You can get the list of vm sizes available in your region by using the CLI command
# 
# ```shell
# az vm list-skus -o tsv
# ```
# * min_nodes - this sets the minimum size of the cluster.  If you set the minimum to 0 the cluster will shut down all nodes while note in use.  Setting this number to a value higher than 0 will allow for faster start-up times, but you will also be billed when the cluster is not in use.
# * max_nodes - this sets the maximum size of the cluster.  Setting this to a larger number allows for more concurrency and a greater distributed processing of scale-out jobs.
# 
# 
# To create a **CPU** cluster now, run the cell below. The autoscale settings mean that the cluster will scale down to 0 nodes when inactive and up to 4 nodes when busy.

# In[20]:


from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

# Choose a name for your CPU cluster
cpu_cluster_name = "cpucluster"

# Verify that cluster does not exist already
try:
    cpu_cluster = ComputeTarget(workspace=ws, name=cpu_cluster_name)
    print("Found existing cpucluster")
except ComputeTargetException:
    print("Creating new cpucluster")
    
    # Specify the configuration for the new cluster
    compute_config = AmlCompute.provisioning_configuration(vm_size="STANDARD_D2_V2",
                                                           min_nodes=0,
                                                           max_nodes=4)

    # Create the cluster with the specified name and configuration
    cpu_cluster = ComputeTarget.create(ws, cpu_cluster_name, compute_config)
    
    # Wait for the cluster to complete, show the output log
    cpu_cluster.wait_for_completion(show_output=True)


# ---
# 
# ## Next steps
# 
# In this notebook you configured this notebook library to connect easily to an Azure ML workspace.  You can copy this notebook to your own libraries to connect them to you workspace, or use it to bootstrap new workspaces completely.
# 
# If you came here from another notebook, you can return there and complete that exercise, or you can try out the [Tutorials](./tutorials) or jump into "how-to" notebooks and start creating and deploying models.  A good place to start is the [train within notebook](./how-to-use-azureml/training/train-within-notebook) example that walks through a simplified but complete end to end machine learning process.

# In[28]:


#Connect to workspace
# load workspace configuration from the config.json file in the current folder.
ws = Workspace.from_config()
print(ws.name, ws.location, ws.resource_group, ws.location, sep='\t')


# In[29]:


#Create experiment
#Create an experiment to track the runs in your workspace. A workspace can have muliple experiments.

experiment_name = 'energydata_linear_regression'
from azureml.core import Experiment
exp = Experiment(workspace=ws, name=experiment_name)


# Create or Attach existing compute resource
# By using Azure Machine Learning Compute, a managed service, data scientists can train machine learning models on clusters of Azure virtual machines. Examples include VMs with GPU support. In this tutorial, you create Azure Machine Learning Compute as your training environment. The code below creates the compute clusters for you if they don't already exist in your workspace.
# 
# Creation of compute takes approximately 5 minutes. If the AmlCompute with that name is already in your workspace the code will skip the creation process.

# In[30]:




from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget
import os

# choose a name for your cluster
compute_name = os.environ.get("AML_COMPUTE_CLUSTER_NAME", "cpucluster")
compute_min_nodes = os.environ.get("AML_COMPUTE_CLUSTER_MIN_NODES", 0)
compute_max_nodes = os.environ.get("AML_COMPUTE_CLUSTER_MAX_NODES", 4)

# This example uses CPU VM. For using GPU VM, set SKU to STANDARD_NC6
vm_size = os.environ.get("AML_COMPUTE_CLUSTER_SKU", "STANDARD_D2_V2")


if compute_name in ws.compute_targets:
    compute_target = ws.compute_targets[compute_name]
    if compute_target and type(compute_target) is AmlCompute:
        print('found compute target. just use it. ' + compute_name)
else:
    print('creating a new compute target...')
    provisioning_config = AmlCompute.provisioning_configuration(vm_size = vm_size,
                                                                min_nodes = compute_min_nodes, 
                                                                max_nodes = compute_max_nodes)

    # create the cluster
    compute_target = ComputeTarget.create(ws, compute_name, provisioning_config)
    
    # can poll for a minimum number of nodes and for a specific timeout. 
    # if no min node count is provided it will use the scale settings for the cluster
    compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)
    
     # For a more detailed view of current AmlCompute status, use get_status()
    print(compute_target.get_status().serialize())

