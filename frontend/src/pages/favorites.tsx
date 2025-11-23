import { SidebarProvider, SidebarInset, SidebarTrigger } from '@/components/ui/sidebar'
import { AppSidebar } from '@/components/AppSidebar'
import { Separator } from '@/components/ui/separator'
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbList,
  BreadcrumbPage,
} from '@/components/ui/breadcrumb'
import { Button } from '@/components/ui/button'
import env from '@/constants'
import { useState } from 'react'

export function Favorites() {
  const [testResponse, setTestResponse] = useState<string>('')
  const testRequest = () => {
    fetch(`${env.API_URL}/api/test`)
      .then(response => {
        if (!response.ok) {
          throw new Error('Network response was not ok')
        }
        return response.json()
      })
      .then(data => setTestResponse(data))
      .catch(error => console.error(`Error: ${error}`))
      .finally(() => console.log('Request completed'))
  }

  return (
    <SidebarProvider>
      <AppSidebar />
      <SidebarInset>
        <header className="flex h-16 shrink-0 items-center gap-2 border-b px-4">
          <SidebarTrigger className="-ml-1" />
          <Separator orientation="vertical" className="mr-2 h-4" />
          <Breadcrumb>
            <BreadcrumbList>
              <BreadcrumbItem>
                <BreadcrumbPage>Favorites</BreadcrumbPage>
              </BreadcrumbItem>
            </BreadcrumbList>
          </Breadcrumb>
        </header>
        <div className="flex flex-1 flex-col gap-4 p-4">
          <div className="grid gap-4">
            <div className="flex items-center justify-between">
              <Button variant="outline" onClick={testRequest}>
                test request
              </Button>
              <p>{testResponse}</p>
            </div>
          </div>
        </div>
      </SidebarInset>
    </SidebarProvider>
  )
}