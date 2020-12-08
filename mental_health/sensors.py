class validate:
    def __init__(self,model_type,input_types,map_):
        for i in map_[model_type]:
            if(i not in input_types):
                raise ValueError(str(i)+": Not found for model type :"+str(model_type))

class get_input:
    import pandas as pd

    function_map={"gps":"import_csv",
                  "activity":"import_csv",
                  "dark":"import_csv",
                  "phonelock":"import_csv",
                  "audio":"import_csv",
                  "call_log":"import_csv",
                  "sms":"import_csv",
                  "conversation":"import_csv"}

    def import_csv(self, input_path):
        return(self.pd.read_csv(input_path))

    def __init__(self,input_paths):
        self.input_types=list(input_paths.keys())
        self.data_files={i:None for i in self.input_types}
        for i in self.input_types:
            function_name=self.function_map[i]
            self.data_files[i]=eval("self."+function_name)(input_paths[i])




