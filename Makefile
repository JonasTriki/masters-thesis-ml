# Change the configuration here.
# Include your useid/name as part of IMAGENAME to avoid conflicts
IMAGENAME = docker-jtr008-masters
CONFIG    = masters
COMMAND   = bash
DISKS     = -v $(PWD)/code:/project
USERID    = $(shell id -u)
GROUPID   = $(shell id -g)
USERNAME  = $(shell whoami)
PORT      = -p 1337:1337 -p 1338:1338
NETWORK   = --network host
# Only use GPU #1.
GPUS      = --gpus device=0
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

.docker: Dockerfile-$(CONFIG)
	docker build $(USERCONFIG) -t $(USERNAME)-$(IMAGENAME) $(NETWORK) -f Dockerfile-$(CONFIG) .

# Using -it for interactive use
RUNCMD=docker run $(GPUS) $(RUNTIME) $(NETWORK) --rm --user $(USERID):$(GROUPID) $(DETACH) $(PORT) $(SSHFSOPTIONS) $(DISKS) -it $(USERNAME)-$(IMAGENAME)

# Replace 'bash' with the command you want to do
default: .docker
	$(RUNCMD) $(COMMAND)

# requires CONFIG=jupyter
jupyter:
	$(RUNCMD) jupyter notebook --ip $(hostname -I) --port 1337
