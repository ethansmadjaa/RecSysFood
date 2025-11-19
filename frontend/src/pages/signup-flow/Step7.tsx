interface StepProps {
  data: Record<string, unknown>
  onUpdate: (data: Record<string, unknown>) => void
}

export function Step7BudgetPreferences(_props: StepProps) {
  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-foreground mb-2">
          Budget Preferences
        </h2>
        <p className="text-muted-foreground">
          What's your typical budget for meals?
        </p>
      </div>

      <div className="space-y-4">
        {/* Add your form fields here */}
        <p className="text-center text-muted-foreground">
          Step 7 content - Budget preferences will go here
        </p>
      </div>
    </div>
  )
}
