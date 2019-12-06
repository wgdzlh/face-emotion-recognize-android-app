/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package edu.hust.ccstudio.ferdemo.tflite;

import android.content.Context;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.util.Log;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.util.List;
import java.util.Map;


public class Classifier {
    private static final String logTag = "Classifier";
    private static final String modelPath = "fer_model.tflite";
    private static final String labelsPath = "labels.txt";

    public enum Device {
        CPU,
        NNAPI,
        GPU
    }

//    private static final int MAX_RESULTS = 3;

    // The loaded TensorFlow Lite model.
    private MappedByteBuffer tfliteModel;

    // Image size along the x axis.
    private final int imageSizeX;

    // Image size along the y axis.
    private final int imageSizeY;

    // Optional GPU delegate for acceleration.
    private GpuDelegate gpuDelegate = null;

    // Optional NNAPI delegate for acceleration.
    private NnApiDelegate nnApiDelegate = null;

    // An instance of the driver class to run model inference with Tensorflow Lite.
    private Interpreter tflite;

    // Labels corresponding to the output of the vision model.
    private List<String> labels;

    // Input image TensorBuffer.
    private TensorBuffer inputImageBuffer;

    // Output probability TensorBuffer.
    private final TensorBuffer outputProbabilityBuffer;

    // Processor to apply post processing of the output probability.
//    private final TensorProcessor probabilityProcessor;

    private final float[] grayScale;
    private final int[] pixels;

    public Classifier(Context context, Device device, int numThreads) throws IOException {
        tfliteModel = FileUtil.loadMappedFile(context, modelPath);
        // Options for configuring the Interpreter.
        Interpreter.Options tfliteOptions = new Interpreter.Options();
        switch (device) {
            case NNAPI:
                nnApiDelegate = new NnApiDelegate();
                tfliteOptions.addDelegate(nnApiDelegate);
                break;
            case GPU:
                gpuDelegate = new GpuDelegate();
                tfliteOptions.addDelegate(gpuDelegate);
                break;
            case CPU:
                break;
        }
        tfliteOptions.setNumThreads(numThreads);
        tflite = new Interpreter(tfliteModel, tfliteOptions);

        // Loads labels out from the label file.
        labels = FileUtil.loadLabels(context, labelsPath);

        // Reads type and shape of input and output tensors, respectively.
        int imageTensorIndex = 0;
        int[] imageShape = tflite.getInputTensor(imageTensorIndex).shape(); // {1, height, width, 1}
        // Log.d(logTag, TextUtils.join(", ", Arrays.stream(imageShape).boxed().collect(Collectors.toList())));

        imageSizeY = imageShape[1];
        imageSizeX = imageShape[2];
        grayScale = new float[imageSizeX * imageSizeY];
        pixels = new int[imageSizeX * imageSizeY];
        DataType imageDataType = tflite.getInputTensor(imageTensorIndex).dataType();
        Log.d(logTag, imageDataType.name());
        int probabilityTensorIndex = 0;
        int[] probabilityShape =
                tflite.getOutputTensor(probabilityTensorIndex).shape(); // {1, NUM_CLASSES}
        // Log.d(logTag, TextUtils.join(", ", Arrays.stream(probabilityShape).boxed().collect(Collectors.toList())));

        DataType probabilityDataType = tflite.getOutputTensor(probabilityTensorIndex).dataType();

        // Creates the input tensor.
        inputImageBuffer = TensorBuffer.createFixedSize(imageShape, imageDataType);

        // Creates the output tensor and its processor.
        outputProbabilityBuffer = TensorBuffer.createFixedSize(probabilityShape, probabilityDataType);

        // Creates the post processor for the output probability.
//        probabilityProcessor = new TensorProcessor.Builder().build();

        Log.d(logTag, "Created a Tensorflow Lite Image Classifier.");
    }

    // Loads input image, and applies preprocessing.
    private void loadImage(final Bitmap original) {
        Bitmap bitmap = Bitmap.createScaledBitmap(original, imageSizeX, imageSizeY, true);
        bitmap.getPixels(pixels, 0, imageSizeX, 0, 0, imageSizeX, imageSizeY);
        int count = 0;
        for (int pixel : pixels) {
            // bitmap is converted to gray scale, reversed and normalized.
            grayScale[count++] = (255f - (
                    0.2989f * ((pixel >> 16) & 0xFF) + 0.5870f * ((pixel >> 8) & 0xFF) + 0.1140f * (pixel & 0xFF)
            )) / 255f;
        }
        inputImageBuffer.loadArray(grayScale);
    }

    // Runs inference and returns the classification results.
    public Map<String, Float> recognizeImage(final Bitmap bitmap) {
        // Logs this method so that it can be analyzed with systrace.
//        Trace.beginSection("recognizeImage");
//        Trace.beginSection("loadImage");
//        long startTimeForLoadImage = SystemClock.uptimeMillis();
        loadImage(bitmap);
//        long endTimeForLoadImage = SystemClock.uptimeMillis();
//        Trace.endSection();
//        Log.v(logTag, "Time cost to load the image: " + (endTimeForLoadImage - startTimeForLoadImage));
        // Runs the inference call.
//        Trace.beginSection("runInference");
        long startTimeForReference = SystemClock.uptimeMillis();
        tflite.run(inputImageBuffer.getBuffer(), outputProbabilityBuffer.getBuffer().rewind());
        long endTimeForReference = SystemClock.uptimeMillis();
//        Trace.endSection();
        Log.v(logTag, "Time cost to run model inference: " + (endTimeForReference - startTimeForReference));

//        TensorBuffer outBuff = probabilityProcessor.process(outputProbabilityBuffer);
//        Trace.endSection();
        // Gets the map of label and probability.
        return new TensorLabel(labels, outputProbabilityBuffer).getMapWithFloatValue();
    }

    // Closes the interpreter and model to release resources.
    public void close() {
        if (tflite != null) {
            tflite.close();
            tflite = null;
        }
        if (gpuDelegate != null) {
            gpuDelegate.close();
            gpuDelegate = null;
        }
        if (nnApiDelegate != null) {
            nnApiDelegate.close();
            nnApiDelegate = null;
        }
        tfliteModel = null;
    }

}
