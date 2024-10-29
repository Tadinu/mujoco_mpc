"""Merge the mpl and panda models."""

import sys

panda_filename = sys.argv[1]
mpl_filename = sys.argv[2]
merge_filename = sys.argv[3]
underactuated = (sys.argv[4] == "1")

with open(panda_filename) as panda_file:
  panda = panda_file.read()
with open(mpl_filename) as mpl_file:
  mpl = mpl_file.read()

# insert defaults
default_begin_index = mpl.index('<default>')  # include default tag
last_default_index = mpl.rindex('</default>')
defaults = mpl[default_begin_index: last_default_index]
panda = panda.replace('<default>', defaults)

# insert assets
asset_begin_index = mpl.index('<asset>')  # include asset tag
asset_close_index = mpl.index('</asset>', asset_begin_index)
assets = mpl[asset_begin_index:asset_close_index]
panda = panda.replace('<asset>', assets)

# attach model
worldbody_index = mpl.index('<worldbody>') + len('<worldbody>')
close_worldbody_index = mpl.index('</worldbody>', worldbody_index)
mpl_body = mpl[worldbody_index:close_worldbody_index]
panda = panda.replace('<site name="attachment_site"/>', mpl_body)

# insert bottom: contact, tendon, equality
contact_begin_index = mpl.index('</worldbody>')  # include closing tag
equality_close_index = mpl.index(
    '</equality>', contact_begin_index) + len('</equality>')
bottom = mpl[contact_begin_index:equality_close_index]
panda = panda.replace('</worldbody>', bottom)

# add gravity compensation to all bodies
panda = panda.replace('<body ', '<body gravcomp="1" ')

# eliminate contact with the target
panda = panda.replace('priority="1"',
                      'priority="1" contype="6" conaffinity="5"')
panda = panda.replace(
    '<geom type="mesh" group="3"/>',
    '<geom type="mesh" group="3" contype="2" conaffinity="1"/>')

# add actuators
if underactuated:
    print("UNDERACTUATED PANDA-MPL")
    mpl_actuator_index = mpl.index('<actuator>')
    mpl_close_actuator_index_index = mpl.index('</actuator>', mpl_actuator_index) + len('</actuator>')
    mpl_actuators = mpl[mpl_actuator_index:mpl_close_actuator_index_index]

    actuator_begin_index = panda.index('<actuator>')
    actuator_close_index = panda.index('</actuator>', actuator_begin_index) + len('</actuator>')
    actuators = panda[actuator_begin_index:actuator_close_index]
    panda = panda.replace(actuators, mpl_actuators)
else:
    print("FULLY-ACTUATED PANDA-MPL")
    mpl_actuator_index = mpl.index('<actuator>') + len('<actuator>')
    mpl_close_actuator_index_index = mpl.index('</actuator>', mpl_actuator_index) + len('</actuator>')
    mpl_actuators = mpl[mpl_actuator_index:mpl_close_actuator_index_index]

    actuator_begin_index = panda.index('</actuator>')
    actuator_close_index = actuator_begin_index + len('</actuator>')
    actuators = panda[actuator_begin_index:actuator_close_index]
    panda = panda.replace(actuators, mpl_actuators)

# remove panda keyframe
keyframe_begin_index = panda.index('<keyframe>')  # keep tag (for removal)
keyframe_close_index = panda.index('</keyframe>') + len('</keyframe>')
panda = panda.replace(panda[keyframe_begin_index:keyframe_close_index], '')

with open(merge_filename, 'w') as merged_file:
  merged_file.write(panda)
