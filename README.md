# BraTS18 segmentation using NVDIAI Flare

This demo shows how to deploy federated training & validation of
BraTS18 segmentation in __real-world deployment scenario__ using NVIDIA
Flare (nvflare).

This demo is based on nvflare's official demo
[here](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), with
slight modification and code repackaging. Contrary to using nvflare
`simulator` or `poc`, which are perfect for local testing, this demo
uses the `provision` method to securely generate different deployment
packages.

Here are the steps to deploy the BraTS18 segmentation model to run
distributed training & validation.

## Evironment set-up

You can install nvflare using different ways (see
[here](https://nvflare.readthedocs.io/en/main/getting_started.html#installation)
for more details). In this demo, we will use the python virtual
environment (here via `virtualenv`, but `venv` will work as well).

Create & activate a virtual environment and install dependencies for
this demo (including nvflare v2.3 and MONAI):
```
virtualenv -p -python3 ./venv
source ./venv/bin/activate

pip install -r requirements.txt
```

Stay inside the virtual environment for the rest of the demo.

## Provision & generate packages for different participants

To provision for your project, simply run:
```
nvflare provision -p ./project.yml
```
Notice that when running `nvlfare provision` without any arguments, the
program will create a default `project.yml` for you, based on a master
template. The `project.yml` file defines configuration of your
project, the participants and different set-ups for the provisioning
process. An example `project-brats18-demo.yml` file is already provided in
 this demofolder.

We need to now modify the `project-brats18-demo.yml` file based on our use-case. The
provided file contains only minimal base-line fields
that are sufficient for this demo. For real deployment, you might want
add more configurations for the provisioning. For more details on
configurations, consult [this documentation
page](https://nvflare.readthedocs.io/en/main/programming_guide/provisioning_system.html#id2).

Here are the fields that we need to modify
- The `participants`
  - Give names to the `server`, the `admin` and the `client`s. You are
    free to add any number of clients, but for this demo we will stay
    with two clients.
  - Noted that the `server` name must be a **fully qualified domain
    name** so it can be accessed from anywhere by clients.
  - For `client` and `admin` names, you're free to set up any semantic
    names.
  - You also need to make sure that the `fed_learn_port` and
    `admin_port` are available. If they are not, you can change them
    to available ports.
- The `sp_end_point`
  - This is the service provider endpoint. For this demo, we can just
    set it to the `server`'s fully qualified domain name, since we are
    not using the **High Availability** mode with backup server. For
    details on **High Availability** mode, consult [this documentation
    page](https://nvflare.readthedocs.io/en/main/programming_guide/high_availability.html).

Once the content of `project-brats-demo.yml` is properly modified, we
can run:
```
nvflare provision -p ./project-brats18-demo.yml
```
This will generate a `workspace/prod_00` folder structure, inside
which a folder is created for each of the participants (`server`,
`admin`, `client`s) defined in the
YAML config file. The name of each folder corresponds to the name
given to each participant. It is also possible to zip and password
protect each generated folder. See details
[here](https://nvflare.readthedocs.io/en/main/real_world_fl/overview.html#customize-the-provision-configuration).

## Package folder structure for each participants

For each participant, a folder with the following sub-folders will be
generated after provision:
- The `startup` sub-folder: this one contains scripts and config files to
  kick start the process for the participant. Here are the important
  scripts that we will interact with.
  - For the `server`, there will be a `start.sh` script that fires up
    the server when launched.
  - For the `client`, there will be a `start.sh` script that allows
    the client to connect to the server when launched.
  - For the `admin`, there will be a script `fl_admin.sh` script that
    allows for opening the administrator console when launched, so
    that the project admin can control & manage the whole project.
  - **Do not modify any content of the `startup` folder**. All
    files under this folder are secured signed for security
    purpose. Any modification of the files will trigger the system to
    disallow the participant to run its startup script.
  - You will not need to modify the content of the `startup` folder
    anyway. The scripts are automatically generated with proper
    configuration based on the project YAML file, so when you launch
    the corresponding script, the `server` will open the right ports,
    and the `client`s will know where is the server to connect to. All
    connections are encrypted.
- The `local` sub-folder: this one contains default site policy files
  generated by provisioning. These policy files can be overridden to
  provide site-specific policies for resources, privacy, authorization
  and logging. For the sake of this demo, we will leave them
  unmodified. But for real-world deployment, it is recommended to set
  them up. Please consult [this documentation
  page](https://nvflare.readthedocs.io/en/main/user_guide/site_policy_management.html)
  regarding site-sepcific policies.
- The `transfer` folder: this one contains will hold artefacts (files,
  codes, ...) that needs to be transmitted between different
  participants. It is especially important for the `admin`
  participant, because this is the default folder to hold the `app`s
  that need to be distributed to `server` and `client`s to run.

For more details regarding folder structures, consult the following
documentation pages:
- [Server folder
  structure](https://nvflare.readthedocs.io/en/main/real_world_fl/workspace.html#server-workspace)
- [Client folder structure](https://nvflare.readthedocs.io/en/main/real_world_fl/workspace.html#client-workspace)
- [Admin folder
  structure](https://nvflare.readthedocs.io/en/main/real_world_fl/overview.html#administrator-side-folder-and-file-structure)

We now need to send the generated folders to the corresponding
participants, so they can execute their scripts and configuration
policies on their hardware dedicated to the project. This can be done
by email, sftp or any other preferred secured file transfer methods.

## Data preparation

The BraTS18 training data can be downloaded from [this kaggle
page](https://www.kaggle.com/datasets/sanglequang/brats2018?select=MICCAI_BraTS_2018_Data_Training)
(MICCAI_BraTS_2018_Data_Training).

For the simplicity of this demo, we will put data for each `client`
under the `/tmp/dataset_brats18` folder. Let's go ahead and create
the following folders for each `client`:
```
mkdir -p /tmp/dataset_brats18/dataset
mkdir -p /tmp/dataset_brats18/datalist
```
Then put the downloaded data (The `HGG`, `LGG` folders and the
`survival_data.csv` file) under `/tmp/dataset_brats18/dataset`.

The `datalist` folder for each `client` is used to hold a json file
for its site-specific data split for training and validation. Example
json files are provided in this demo:
```
datalist/site-1.json
datalist/site-2.json
```
Copy them under the corresponding `client`'s
`/tmp/dataset_brats18/datalist` folder. Do not forget to modify the
file's name to match the `client`'s name defined during provisioning.

This way of data arrangement is only to simplify the demo. You have the
flexibility to arrange data any way you want. And in real-world case,
you probably will have site-specific data structure. You can configure
nvflare to ingest data based on each `client` site's
structure, for example, by providing site-specific data structure
description files to the `Trainer` during runtime.

## Application for federated deployment

The last piece of this demo is the actual application, a.k.a the
BraTS18 segmentation application that we want to distribute to the
server and clients to run it's training and validation in a federated
fashion.

This application is packaged inside the `brats_fedavg_app` folder
based on nvflare's application folder structure, described as
follows:
- A `meta.json` config file, which can be used to configure high level
  info of the app. It can also be used to define to which participants
  this app should be sent to.
- An `app` folder which holds the actual application, inside which we
  have:
  - A `custom` folder, which contains your custom codes. You are free
    to arrange the structure of your codes in any way you prefer. What
    is important is to write corresponding training and validation
    codes for the `server` and the `client`s. This can be done quite
    easily using or extending nvflare's built-in classes and
    APIs. Please refer to the programming guide
    [here](https://nvflare.readthedocs.io/en/main/programming_guide.html). I
    also recommend you look into the example codes provided in this
    demo to learn more. One thing to notice here is, during runtime,
    the system will add this `custom` folder to the `PYTHONPATH`, so
    you can import any content under this folder inside codes using
    the usaual python `import`.
  - A `config` folder, which essentially contains runtime & workflow
    specific configurations for the `server` and the `client`s.
    - The `config_fed_client.json` file defines runtime specific
      configs for the `client`. For example, how to instantiate the
      the `Trainer`, by referring to the path of the
      `SupervisedMonaiBratsLearner` class under the `custom` folder.
    - The `config_fed_server.json` file defines runtime specific
      configs for the `server`. In this demo, we're using the `Scatter
      and Gather` workflow, which is nvflare's implementation of the
      `fedavg` workflow. You can find `server`-side hyperparameters
      for the `fedavg` workflow such as total number of rounds, epochs
      per clients etc.
    - You are also free to add any other runtime config files
      here. For instance, in this demo we added a `config_train.json`
      to be read by the `Trainer`,
      that contains several hyperparameters for training &
      aggregation, as well as dataset location definitions. Notice
      that we can also use a custom config file here to feed
      site-specific data structure description to the `Trainer`, so
      that the workflow can ingest each `client` site's data with
      specific structure.

The last thing to do before running the project, is to send this
application folder under the `transfer` sub-folder inside the `admin`'s
folder generated during provisioning. Doing this will allow the
`admin` user to broadcast this application to `server` and `client`s
for distributed training & validation.

## Start the server, connect clients, submit job

Now we can go ahead and start up the project.

First, we need to fire up the `server`. On the `server` side, execute
the `startup.sh` script:
```
<PATH TO THE PROVISION GENERATED SERVER FOLDER>/startup/start.sh
```
After some time, you should see a log similar to:
```
2023-08-09 09:18:28,245 - root - INFO - Server started
```

Then, connect all `client`s to the server. On each `client`'s side, execute:
```
<PATH TO THE PROVISION GENERATED CLIENT FOLDER>/startup/start.sh
```

After sometime, you should see a log on each `client`'s side similar
to:
```
2023-08-09 11:42:47,439 - FederatedClient - INFO - Successfully
registered client:site-1 for project XXX. Token:xxxx
```

Now, let connect as the `admin` user. You can connect as `admin` from
anywhere where you have access to the provision generated `admin`
folder, by executing:
```
<PATH TO THE PROVISION GENERATED ADMIN FOLDER>/startup/fl_admin.sh
```

When connected as admin, you will a console prompt similar to:
```
Logged into server at XXX:XXX with SSID: XXXX
Type ? to list commands; type "? cmdName" to show usage of a command.
>
```

This is the `admin` console. It allows the `admin` user to control &
manage the whole project, performing tasking such as:
- Querrying the server / client status with command: `check_status
  server / client`.
- Submitting / Aborting jobs with: `submit_job <APP NAME>` /
  `abort_job <JOB ID>`.
- Listing jobs with: `list_job`
- Shutting down clients / server with: `shutdown client / server`
- ...
For a comprehensive list of `admin` commands, refer to the
documentation
[here](https://nvflare.readthedocs.io/en/main/real_world_fl/operation.html). Notice
that, it is also possible to perform `admin` tasks in a programmatic
way [using
APIs](https://nvflare.readthedocs.io/en/main/real_world_fl/flare_api.html).

Now we will submit the application to launch distributed training &
validation. We do this via the `submit_job` command in the `admin`
console:
```
submit_job brats_fedavg_app
```
SInce we've already sent the application to the `transfer` sub-folder
under `admin`'s folder, the `admin` will recognize this application by
its folder name, and then broadcast the application to the `server`
and `client` for distributed training & validation.

After some time, you will see training logs inside both the `server`
and `client` consoles. You can also querry the individual `log.txt` generated
under the `server` and `client` folder. For a more visual tracking of
the training process, it is also possible to log events using
tensorboard and then visualize using the tensorboard program (see
details
[here](https://nvflare.readthedocs.io/en/main/examples/tensorboard_streaming.html)).
