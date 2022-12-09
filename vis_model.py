import torch
SHAPE = 224                                       # your batch shape
OUT_PATH = "output/models/onnx"                                    # output path
x = torch.randn(SHAPE)
with torch.no_grad():
    # extract the module from dataparallel models
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.cpu()
    # the converter works best on models stored on the CPU
    model.eval()
    torch.onnx.export(model,                      # model being run
                      # model input (or a tuple for multiple inputs)
                      x,
                      # where to save the model (can be a file or file-like object)
                      OUT_PATH,
                      export_params=True,         # store the trained parameter weights inside the model
                      opset_version=11)           # it's best to specify the opset version. At time of writing 11 was the latest release
