OBJS = Utils.cu ImageUtils.cu SceneUtils.cu CameraUtils.cu Rays.cu Render.cu VolumeFieldRender.cu Optimizer.cu RenderUtils.cu

target := VFR

CC = nvcc

$(target):$(OBJS)
	$(CC)  $(OBJS) -o $(target) -rdc=true -lcurand -w

.PHONY:clean
clean:
	-rm -rf $(target)
