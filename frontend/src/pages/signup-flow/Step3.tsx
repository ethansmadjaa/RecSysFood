interface StepProps {
  data: Record<string, unknown>
  onUpdate: (data: Record<string, unknown>) => void
}

export function Step3Allergies(_props: StepProps) {
  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-foreground mb-2">
          Allergies & Restrictions
        </h2>
        <p className="text-muted-foreground">
          Let us know about any food allergies or restrictions
        </p>
      </div>

      <div className="space-y-4">
        {/* Add your form fields here */}
        <p className="text-center text-muted-foreground">
          Step 3 content - Allergies and restrictions form will go here
        </p>
      </div>
    </div>
  )
}
