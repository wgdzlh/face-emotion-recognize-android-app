package edu.hust.ccstudio.ferdemo.util;

import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;


public class RectOverlay extends GraphicOverlay.Graphic {
    private static final Paint mRectPaint = new Paint();
    private static final Paint mTextPaint = new Paint();
    private final Rect mRect;
    private final String mLabel;

    static {
        mRectPaint.setColor(Color.GREEN);
        mRectPaint.setStyle(Paint.Style.STROKE);
        mRectPaint.setStrokeWidth(3f);

//        mTextPaint.setColor(Color.MAGENTA);
        mTextPaint.setTextSize(100f);
    }

    public RectOverlay(GraphicOverlay overlay, Rect rect, String label) {
        super(overlay);
        mRect = rect;
        mLabel = label;
    }

    @Override
    public void draw(Canvas canvas) {
        canvas.drawRect(mRect, mRectPaint);
        canvas.drawText(mLabel, mRect.left, mRect.top, mTextPaint);
    }
}
