package edu.hust.ccstudio.ferdemo;

import android.app.AlertDialog;
import android.graphics.Bitmap;
import android.graphics.Rect;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.util.Log;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import com.google.firebase.ml.vision.FirebaseVision;
import com.google.firebase.ml.vision.common.FirebaseVisionImage;
import com.google.firebase.ml.vision.face.FirebaseVisionFace;
import com.google.firebase.ml.vision.face.FirebaseVisionFaceDetector;
import com.google.firebase.ml.vision.face.FirebaseVisionFaceDetectorOptions;
import com.wonderkiln.camerakit.CameraKitEventCallback;
import com.wonderkiln.camerakit.CameraKitImage;
import com.wonderkiln.camerakit.CameraView;

import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import butterknife.BindView;
import butterknife.ButterKnife;
import butterknife.OnClick;
import dmax.dialog.SpotsDialog;
import edu.hust.ccstudio.ferdemo.tflite.Classifier;
import edu.hust.ccstudio.ferdemo.util.GraphicOverlay;
import edu.hust.ccstudio.ferdemo.util.RectOverlay;


public class EmotionDetector extends AppCompatActivity implements CameraKitEventCallback<CameraKitImage> {

    @BindView(R.id.camera_view)
    CameraView mCameraView;
    @BindView(R.id.graphic_overlay)
    GraphicOverlay mGraphicOverlay;
    private AlertDialog mAlertDialog;
    private FirebaseVisionFaceDetector mFirebaseVisionFaceDetector;
    private Classifier emotionClassifier;
    private HandlerThread handlerThread;
    private Handler handler;
    private long startTimeForReference;

    private static final String logTag = "EMO_DETECT";

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_emotion_detector);
        ButterKnife.bind(this);
        setTitle(R.string.emotion_detector);

        try {
            emotionClassifier = new Classifier(this, Classifier.Device.GPU, 4);
        } catch (IOException e) {
            Log.e(logTag, e.getMessage());
        }

        FirebaseVisionFaceDetectorOptions defaultOpts =
                new FirebaseVisionFaceDetectorOptions.Builder().build();
        mFirebaseVisionFaceDetector = FirebaseVision.getInstance()
                .getVisionFaceDetector(defaultOpts);

        // important! set this to match the preview (mCameraView) size and the actual image size.
        mCameraView.setCropOutput(true);
    }

    @Override
    public void callback(CameraKitImage cameraKitImage) {
        mCameraView.stop();
        Log.i(logTag, "time passed after capturing image: "
                + (SystemClock.uptimeMillis() - startTimeForReference));
        runInBackground(() -> {
            Bitmap original = cameraKitImage.getBitmap();
            Bitmap bitmap = Bitmap.createScaledBitmap(original,
                    mCameraView.getWidth(), mCameraView.getHeight(), true);
            original.recycle();
//            Log.i(logTag, String.format("Image size: %d, %d", bitmap.getWidth(), bitmap.getHeight()));
            processFaceDetection(bitmap);
        });
    }

    @OnClick(R.id.detect_emotion_btn)
    public void startDetect() {
        if (mCameraView.isStarted()) {
            runInBackground(mAlertDialog::show);
            startTimeForReference = SystemClock.uptimeMillis();
            mCameraView.captureImage(this);
        } else {
            Toast.makeText(this, R.string.please_start_over, Toast.LENGTH_SHORT).show();
        }
    }

    @OnClick(R.id.toggle_camera_btn)
    public void toggle_camera() {
        startOver();
        mCameraView.toggleFacing();
    }

    @OnClick(R.id.start_over_btn)
    public void startOver() {
        mGraphicOverlay.clear();
        mCameraView.start();
    }

    private void processFaceDetection(Bitmap bitmap) {
        FirebaseVisionImage firebaseVisionImage = FirebaseVisionImage.fromBitmap(bitmap);

        mFirebaseVisionFaceDetector.detectInImage(firebaseVisionImage)
                .addOnSuccessListener(firebaseVisionFaces ->
                        runInBackground(() -> getFaceResults(firebaseVisionFaces, bitmap)))
                .addOnFailureListener(e ->
                        Toast.makeText(this, "Error: " + e.getMessage(), Toast.LENGTH_SHORT).show());
    }

    private void getFaceResults(List<FirebaseVisionFace> firebaseVisionFaces, Bitmap bitmap) {
        Log.i(logTag, "time passed after detecting faces: "
                + (SystemClock.uptimeMillis() - startTimeForReference));
        for (FirebaseVisionFace face : firebaseVisionFaces) {
            Rect rect = face.getBoundingBox();
            int x, y, w, h;
            x = Math.max(rect.left, 0);
            y = Math.max(rect.top, 0);
            w = Math.min(rect.right, bitmap.getWidth()) - x;
            h = Math.min(rect.bottom, bitmap.getHeight()) - y;
            // recognize facial expressions
            Map<String, Float> classifyResult = emotionClassifier.recognizeImage(
                    Bitmap.createBitmap(bitmap, x, y, w, h));
            Log.i(logTag, classifyResult.toString());
            String label = Collections.max(classifyResult.entrySet(), Map.Entry.comparingByValue()).getKey();
            RectOverlay rectOverlay = new RectOverlay(mGraphicOverlay, rect, label);
            mGraphicOverlay.add(rectOverlay);
        }
        bitmap.recycle();
//        Log.i(logTag, Looper.myLooper() == Looper.getMainLooper() ? "in main thread." : "in background thread.");
        Log.i(logTag, "total time used in this try: "
                + (SystemClock.uptimeMillis() - startTimeForReference));
        mAlertDialog.dismiss();
    }

    @Override
    protected void onResume() {
        super.onResume();
        mCameraView.start();
        handlerThread = new HandlerThread("detector");
        handlerThread.start();
        handler = new Handler(handlerThread.getLooper());
        runInBackground(() -> mAlertDialog = new SpotsDialog.Builder()
                .setContext(this)
                .setMessage(R.string.in_processing)
                .setCancelable(false)
                .build());
    }

    @Override
    protected void onPause() {
        handlerThread.quitSafely();
        try {
            handlerThread.join();
            handlerThread = null;
            handler = null;
        } catch (InterruptedException e) {
            Log.e(logTag, e.getMessage());
        }
        super.onPause();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        try {
            mFirebaseVisionFaceDetector.close();
        } catch (IOException e) {
            Log.e(logTag, e.getMessage());
        }
        emotionClassifier.close();
    }

    private synchronized void runInBackground(final Runnable r) {
        if (handler != null) {
            handler.post(r);
        }
    }

}
