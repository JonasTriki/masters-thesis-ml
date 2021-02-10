# Change the configuration here.
# Include your useid/name as part of IMAGENAME to avoid conflicts
IMAGENAME = docker-jtr008-masters
COMMAND   = bash
DISKS     = -v $(PWD)/code:/project
USERID    = $(shell id -u)
GROUPID   = $(shell id -g)
USERNAME  = $(shell whoami)
PORT      = -p 1337:1337 -p 1338:1338
NETWORK   = --network host
CONTNAME  = --name jtr008-docker-container
RUNTIME   =
# --runtime=nvidia
# No need to change anything below this line

# Allows you to use sshfs to mount disks
SSHFSOPTIONS = --cap-add SYS_ADMIN --device /dev/fuse

USERCONFIG   = --build-arg user=$(USERNAME) --build-arg uid=$(USERID) --build-arg gid=$(GROUPID)

# Check for detached mode
ifeq ($(DETACHED), 1)
	DETACH = -d
else
	DETACH =
endif

.docker: Dockerfile
	docker build $(USERCONFIG) -t $(USERNAME)-$(IMAGENAME) $(NETWORK) -f Dockerfile .

# Using -it for interactive use
RUNCMD=docker run $(RUNTIME) $(NETWORK) --rm --user $(USERID):$(GROUPID) $(DETACH) $(PORT) $(SSHFSOPTIONS) $(DISKS) $(CONTNAME) -it $(USERNAME)-$(IMAGENAME)

# Replace 'bash' with the command you want to do
default: .docker
	$(RUNCMD) $(COMMAND)

# requires CONFIG=jupyter
jupyter:
	$(RUNCMD) jupyter notebook --ip $(hostname -I) --port 1337
