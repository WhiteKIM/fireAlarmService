from pydantic import BaseModel

"""
# Video Input Source Model
# name : Source Name
# source : Source Url, Ip Address..
# describe : Add Some Comment
# location : Sourve View Location
"""
class InputSource(BaseModel):
    name : str
    source : str
    describe : str
    location : str